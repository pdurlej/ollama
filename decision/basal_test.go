package decision

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

// testdata/basal_golden.json holds prompts rendered by the reference
// implementation (basal.server.to_items + basal.prompt.render with the
// Hugging Face chat template) for both option orders of each question.
func TestBasalMatchesReferencePrompts(t *testing.T) {
	raw, err := os.ReadFile("testdata/basal_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Request json.RawMessage     `json:"request"`
		Prompts map[string][]string `json:"prompts"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	var req Request
	if err := json.Unmarshal(golden.Request, &req); err != nil {
		t.Fatal(err)
	}
	req.Model = "basal"
	c, err := CompileWithEncoder(req, "basal")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Request.Rows) != 2*len(c.fields) {
		t.Fatalf("want two rows per question, got %d rows for %d questions", len(c.Request.Rows), len(c.fields))
	}
	for i, f := range c.fields {
		got := []string{c.Request.Rows[2*i].Prompt, c.Request.Rows[2*i+1].Prompt}
		slices.Sort(got)
		want := slices.Sorted(slices.Values(golden.Prompts[f.Name]))
		if len(want) != 2 {
			t.Fatalf("no golden prompts for %q", f.Name)
		}
		for j := range want {
			if got[j] != want[j] {
				t.Errorf("%s prompt %d differs from the reference\n got: %q\nwant: %q", f.Name, j, got[j], want[j])
			}
		}
		for _, row := range c.Request.Rows[2*i : 2*i+2] {
			if !strings.HasSuffix(row.Prompt, `{"answer": "`) || len(row.Candidates) != len(f.Choices) || row.Candidates[0] != "A" {
				t.Errorf("%s: bad row %q %v", f.Name, row.Prompt[len(row.Prompt)-20:], row.Candidates)
			}
		}
	}
}

func TestBasalAveragesOrdersAndCalibrates(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"basal","state":"x","questions":{
		"q":{"type":"choice","instructions":"Pick.","criteria":{"a":null,"b":null,"c":null}},
		"n":{"type":"noul","instructions":"Yes?"}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := CompileWithEncoder(req, "basal")
	if err != nil {
		t.Fatal(err)
	}
	ln := func(p ...float64) []float32 {
		out := make([]float32, len(p))
		for i, v := range p {
			out[i] = float32(math.Log(v))
		}
		return out
	}
	// choice: original order a,b,c = .6,.3,.1; reversed order c,b,a = .2,.2,.6 → a .6, b .25, c .15
	// noul (choices false,true): original .2,.8; reversed true,false = .6,.4 → false .3, true .7
	result := llm.ScoreResponse{Logits: [][]float32{ln(.6, .3, .1), ln(.2, .2, .6), ln(.2, .8), ln(.6, .4)}}
	resp, err := c.Answer("basal", result)
	if err != nil {
		t.Fatal(err)
	}
	q, _ := resp.Answers.Get("q")
	probs := q.(ChoiceAnswer).Probabilities
	for key, want := range map[string]float64{"a": .6, "b": .25, "c": .15} {
		if got, _ := probs.Get(key); math.Abs(got-want) > 1e-6 {
			t.Errorf("P(%s) = %v, want %v", key, got, want)
		}
	}
	n, _ := resp.Answers.Get("n")
	if got := n.(NoulAnswer).Noul; math.Abs(got-.7) > 1e-6 {
		t.Errorf("P(true) = %v, want 0.7", got)
	}

	c.SetTemperatures(map[string]float64{"noul": 2})
	resp, err = c.Answer("basal", result)
	if err != nil {
		t.Fatal(err)
	}
	n, _ = resp.Answers.Get("n")
	want := math.Sqrt(.7) / (math.Sqrt(.7) + math.Sqrt(.3))
	if got := n.(NoulAnswer).Noul; math.Abs(got-want) > 1e-6 {
		t.Errorf("calibrated P(true) = %v, want %v", got, want)
	}
}

func TestBasalRejectsTooManyOptions(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"basal","state":"x","questions":{"q":{"type":"score","instructions":"Level?",
		"criteria":["0","1","2","3","4","5","6","7","8","9","10"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	if _, err := CompileWithEncoder(req, "basal"); err == nil || !strings.Contains(err.Error(), "supports 2–10") {
		t.Fatalf("want an option-limit error, got %v", err)
	}
}

func TestBasalTextDecodesEscapes(t *testing.T) {
	got, err := basalText(json.RawMessage(`{"text":"za\u017c\u00f3\u0142\u0107","path":"a\/b","n":1.50,"tags":[1,true,null]}`))
	if err != nil {
		t.Fatal(err)
	}
	// json.dumps(..., ensure_ascii=False) in the reference, except the number spelling (kept as sent)
	if want := `{"text": "zażółć", "path": "a/b", "n": 1.50, "tags": [1, true, null]}`; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

func TestCalibrationSmallTemperatureIsStable(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"m","state":"x","questions":{"q":{"type":"choice","instructions":"Pick.","criteria":{"a":null,"b":null}}}}`), &req); err != nil {
		t.Fatal(err)
	}
	for _, encoding := range []string{"", "basal"} {
		c, err := CompileWithEncoder(req, encoding)
		if err != nil {
			t.Fatal(err)
		}
		c.SetTemperatures(map[string]float64{"choice": 1e-4})
		logits := make([][]float32, len(c.Request.Rows))
		for i := range logits {
			logits[i] = []float32{0, 0}
		}
		resp, err := c.Answer("m", llm.ScoreResponse{Logits: logits})
		if err != nil {
			t.Fatal(err)
		}
		q, _ := resp.Answers.Get("q")
		if a, _ := q.(ChoiceAnswer).Probabilities.Get("a"); math.Abs(a-.5) > 1e-9 {
			t.Fatalf("%q: P(a) = %v, want 0.5", encoding, a)
		}
	}
}
