.PHONY: test lint build goldens tables

test:
	CGO_ENABLED=0 go test ./...

lint:
	CGO_ENABLED=0 go vet ./...
	golangci-lint run

build:
	CGO_ENABLED=0 go build ./...

# Regenerates Leptonica goldens. Requires leptonica + a C compiler.
# Manual step — never run in CI. Commit the results.
goldens:
	$(MAKE) -C testdata/golden/gen

# Regenerates the activation tables from a Tesseract checkout. Manual step;
# commit the result. TESS=/path/to/tesseract make tables
# gen.go carries //go:build ignore, so the recipe names the FILE, not the dir.
tables:
	go run ./internal/nn/gen/gen.go -src $(TESS)/src/lstm/functions.cpp -out internal/nn/tables.go
