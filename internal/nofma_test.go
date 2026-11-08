package internal

import (
	"bytes"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"testing"
)

// fusedOp matches arm64 fused multiply-add and multiply-subtract mnemonics in
// `go build -gcflags=-S` output, next to the "(file.go:line)" that emitted it.
var fusedOp = regexp.MustCompile(`\(([^()\s]+\.go):\d+\)\s+FN?M(?:ADD|SUB)[DS]\b`)

// TestNoFusedOpsInReferences cross-compiles for linux/arm64 and fails if any
// file in the module compiles to a fused multiply-add. Go fuses x*y+z there
// (not on amd64), which changes float64 results in the last bit. Wrap the
// product in an explicit conversion, float64(x*y), to prevent it.
func TestNoFusedOpsInReferences(t *testing.T) {
	if testing.Short() {
		t.Skip("full arm64 rebuild; skipped with -short")
	}
	allowed := map[string]bool{}
	raw, err := os.ReadFile("nofma_allow.txt")
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range strings.Split(string(raw), "\n") {
		if l = strings.TrimSpace(l); l != "" && !strings.HasPrefix(l, "#") {
			path, reason, ok := strings.Cut(l, "\t")
			if !ok || strings.TrimSpace(reason) == "" {
				t.Fatalf("nofma_allow.txt: %q needs <path><TAB><reason>", l)
			}
			allowed[path] = true
		}
	}

	root, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}").Output()
	if err != nil {
		t.Fatal(err)
	}
	dir := strings.TrimSpace(string(root))
	cmd := exec.Command("go", "build", "-a", "-gcflags=github.com/dobbo-ca/cadmus/...=-S", "./...")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=arm64", "CGO_ENABLED=0")
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		t.Fatalf("arm64 build: %v\n%s", err, out.String())
	}

	found := map[string]bool{}
	for _, m := range fusedOp.FindAllStringSubmatch(out.String(), -1) {
		found[strings.TrimPrefix(m[1], dir+"/")] = true
	}
	for f := range found {
		if !allowed[f] {
			t.Errorf("%s compiles to a fused multiply-add on arm64; wrap the product in float64(...) or allowlist it in nofma_allow.txt with a reason", f)
		}
	}
}
