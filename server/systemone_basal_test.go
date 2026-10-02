package server

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

type basalScoreRunner struct {
	mockRunner
	request llm.ScoreRequest
}

func (r *basalScoreRunner) Score(_ context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	r.request = input
	// original order (false, true) = .2/.8, reversed order (true, false) = .6/.4
	return llm.ScoreResponse{Logits: [][]float32{
		{float32(math.Log(.2)), float32(math.Log(.8))},
		{float32(math.Log(.6)), float32(math.Log(.4))},
	}}, nil
}

// A GGUF with decision.type "basal" is scored in both option orders with basal's
// own prompt, and decision.temperature.* from its metadata calibrates the answer.
func TestSystemOneHandlerBasalEncoding(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	_, digest := createBinFile(t, gguftest.KV{
		"general.architecture":             "llama",
		"llama.decision.type":              "basal",
		"llama.decision.temperature.noul":  float32(2),
		"llama.decision.temperature.score": float32(0.5),
	}, nil)
	configLayer, err := createConfigLayer(model.ConfigV2{ModelFormat: "gguf", ModelFamily: "llama", Capabilities: []string{"decision"}})
	if err != nil {
		t.Fatal(err)
	}
	params, err := manifest.NewLayer(strings.NewReader(`{"num_ctx":2048}`), "application/vnd.ollama.image.params")
	if err != nil {
		t.Fatal(err)
	}
	layers := []manifest.Layer{{MediaType: "application/vnd.ollama.image.model", Digest: digest}, params}
	if err := manifest.WriteManifest(model.ParseName("basal-decision"), *configLayer, layers); err != nil {
		t.Fatal(err)
	}

	runner := &basalScoreRunner{}
	ref := &runnerRef{llama: runner, refCount: 1, sessionDuration: time.Hour}
	s := newServerWithMockRunner(t, &runner.mockRunner)
	s.sched.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
		runner.contextLength = req.opts.NumCtx
		ref.model = req.model
		ref.modelKey = schedulerModelKey(req.model)
		s.sched.loadedMu.Lock()
		s.sched.loaded[ref.modelKey] = ref
		s.sched.loadedMu.Unlock()
		req.successCh <- ref
		return false
	}
	router := gin.New()
	router.POST("/v1/systemone", s.SystemOneHandler)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(
		`{"model":"basal-decision","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body)
	}

	rows := runner.request.Rows
	if len(rows) != 2 || len(runner.request.Fields) != 0 {
		t.Fatalf("want two candidate-scoring rows, got %+v", runner.request)
	}
	for i, wantOptions := range []string{"A. No\nB. Yes", "A. Yes\nB. No"} {
		p := rows[i].Prompt
		if !strings.HasPrefix(p, "<s><|im_start|>system\nYou evaluate the state") ||
			!strings.Contains(p, "State:\nrefund please\n\nQuestion: Refund requested?\nOptions:\n"+wantOptions+"\n\n") ||
			!strings.HasSuffix(p, "<|im_start|>assistant\n{\"answer\": \"") {
			t.Fatalf("row %d is not basal's prompt: %q", i, p)
		}
		if strings.Join(rows[i].Candidates, ",") != "A,B" {
			t.Fatalf("row %d candidates = %v", i, rows[i].Candidates)
		}
	}
	if runner.request.MaxTokens != 2048 {
		t.Fatalf("scoring context = %d", runner.request.MaxTokens)
	}

	var response struct {
		Answers map[string]struct {
			Noul float64 `json:"noul"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	// both orders average to P(true) = .7; temperature 2 gives sqrt(.7) / (sqrt(.7) + sqrt(.3))
	want := math.Sqrt(.7) / (math.Sqrt(.7) + math.Sqrt(.3))
	if got := response.Answers["refund"].Noul; math.Abs(got-want) > 1e-5 {
		t.Fatalf("P(true) = %v, want %v", got, want)
	}
}
