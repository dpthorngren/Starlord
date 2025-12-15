# Project Information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'Starlord'
copyright = '2025, Daniel Thorngren'
author = 'Daniel Thorngren'
release = '0.1.6'

# General Configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
exclude_patterns = []

# Extension Settings
autodoc_typehints = 'description'
autoclass_content = 'class'
autodoc_class_signature = "separated"
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'member-order': 'groupwise',
}

# HTML Output Options
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_sidebars = {"**": ['searchbox.html', 'globaltoc.html']}
