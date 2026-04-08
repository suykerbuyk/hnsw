.DEFAULT_GOAL := help

PREFIX ?= $(HOME)/.local

##@ General
.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Quick start:  make build && make test"

##@ Build
.PHONY: build
build: ## Build all packages
	go build ./...

##@ Test
.PHONY: test
test: ## Run unit tests (skip integration)
	go test -short -cover ./...

.PHONY: integration
integration: ## Run all tests including integration
	go test -cover -count=1 ./...

.PHONY: bench
bench: ## Run benchmarks
	go test -bench=. -benchmem -run='^$$' ./...

##@ Quality
.PHONY: vet
vet: ## Run go vet
	go vet ./...

.PHONY: check
check: vet test ## Run vet + unit tests

##@ Install
.PHONY: install
install: build ## Build and install to PREFIX
	go install ./...

##@ Clean
.PHONY: clean
clean: ## Remove build artifacts
	go clean ./...
