OUT_DIR = out
OUT_PATH := $(OUT_DIR)/diploma

.PHONY: build
build:
	latexmk -xelatex -synctex=1 -jobname=$(OUT_PATH) main.tex

.PHONY: view
view:
	xdg-open $(OUT_PATH).pdf &


.PHONY: prep
prep:
	# install latex
	sudo apt install texlive-base texlive-latex-extra \
		texlive-xetex texlive-lang-cyrillic latexmk texlive-fonts-extra \
		texlive-science texlive-latex-recommended
	# install TimesNewRoman
	sudo apt install ttf-mscorefonts-installer
	# install hunspell with russian dictionary
	sudo apt install hunspell hunspell-ru


HUN_SPELL_DICTS := en_US,ru_RU
HUN_SPELL_IGNORED_FILES = 'preamble'

.PHONY: check
spellcheck:
	@find . -name '*.tex' | \
	grep --invert-match --extended-regexp $(HUN_SPELL_IGNORED_FILES) | \
	xargs hunspell -d $(HUN_SPELL_DICTS) -l -t


.PHONY: clean
clean:
	find . -regextype posix-extended \
		-regex '.+\.(aux|bbl|blg|fdb_latexmk|fls|log|synctex.gz|toc|xdv)' | \
	xargs --no-run-if-empty rm --force
