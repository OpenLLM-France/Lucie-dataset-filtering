# .SILENT: allows to silent command as @
# .include: allows to include other files in makefile such as .env
.PHONY: install test source target help

.DEFAULT_GOAL := help


COM_COLOR   = \033[0;34m
OBJ_COLOR   = \033[0;36m
OK_COLOR    = \033[0;32m
ERROR_COLOR = \033[0;31m
WARN_COLOR  = \033[0;33m
NO_COLOR    = \033[m


CURRENT_DIR := $(shell pwd)
VAR1="VAR1_VALUE"

ifdef VAR2
	VAR2=$(VAR2)
else
	VAR2="VAR2_VALUE"
endif

install: ## Install dependencies
	@echo "$(COM_COLOR)Installing dependencies...$(NO_COLOR)"
	@echo "$(OK_COLOR)Dependencies installed !$(NO_COLOR)"

test: install ## Run tests
	@echo "Running tests..."
	@echo "Tests passed for $(VAR1) and $(VAR2)!"
	@echo "CURRENT_DIR: $(CURRENT_DIR)"

source: target ## Demonstrate the use of automatic variables
	@echo "source_file $< target_file $@"

# demo/%.jpg: /%.jpg
# 	@echo "source_file $< target_file $@"

# SRC=$(subst images/raw/,images/optimized/,$(wildcard images/raw/*.jpg))

# images/optimized/%.jpg: images/raw/%.jpg
#     mkdir -p images/optimized
#     guetzli --quality 85 --verbose $< $@

# images: $(SRC)

# You can specify a number of parallel jobs with make -j<number of jobs>

help:
	@grep -E '(^[a-zA-Z_-]+:.*?##.*$$)|(^##)' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
			/^[a-zA-Z_-]+:.*?##/ {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}; \
			/^##/ {printf "\n\033[33m%s\033[0m\n", substr($$0, 4)}'

