.PHONY: test lint build goldens

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
