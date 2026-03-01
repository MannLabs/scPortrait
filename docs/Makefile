# Minimal makefile for Sphinx documentation
#

github:
	@$(MAKE) html

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
PYTHON        ?= python
SPHINXBUILD   ?= $(PYTHON) -m sphinx.cmd.build
SOURCEDIR     = .
BUILDDIR      = ./_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean github Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
		rm -rf $(SOURCEDIR)/auto_examples
		rm -rf $(BUILDDIR)/html/*
		rm -rf $(BUILDDIR)/latex/*
		rm -rf $(BUILDDIR)/doctrees/*
