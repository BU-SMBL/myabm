
import os, sys, re
import matplotlib
import matplotlib.pyplot as plt
import mymesh

project = 'MyABM'
copyright = '2023, Timothy O. Josephson'
author = 'Timothy O. Josephson'
version = os.environ.get('SPHINX_VERSION', 'dev')
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 
            'sphinx.ext.doctest', 
            'sphinx.ext.todo', 
            'sphinx.ext.mathjax', 
            'sphinx.ext.ifconfig', 
            'sphinx.ext.viewcode', 
            'sphinx.ext.githubpages', 
            'sphinx.ext.napoleon', 
            'sphinx.ext.autosectionlabel', 
            'sphinx.ext.autosummary', 
            'matplotlib.sphinxext.plot_directive',
            'sphinx_design',
            'sphinx.ext.graphviz',
            'sphinx_copybutton',
            'sphinxcontrib.bibtex',
            'jupyter_sphinx',
            'sphinx_gallery.gen_gallery', 
            'sphinx.ext.intersphinx',
            'sphinx_tabs.tabs',
            'pyvista.ext.plot_directive',
            'pyvista.ext.viewer_directive']
autodoc_mock_imports = []
autodoc_member_order = 'bysource'
autosummary_generate = True

templates_path = ['_templates']

from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
sphinx_gallery_conf = {
     'filename_pattern': re.escape(os.sep) + 'demo_',
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'examples',  # path to where to save gallery generated output
     'download_all_examples': False,
     'remove_config_comments': True,
     'capture_repr': (),
     'image_scrapers': ('matplotlib', DynamicScraper()),
     'abort_on_example_error': True,
     'doc_module': 'myabm'
}  

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_title = f"{project} v{release} Manual"
html_theme =  'pydata_sphinx_theme' #'sphinx_rtd_theme' #
html_static_path = ['_static']
html_logo = '_static/myabm_logo.png'
html_css_files = ['css/myabm.css']
pygments_light_style="tango"
pygments_dark_style="nord"
html_theme_options = dict(collapse_navigation=True, 
                           navigation_depth=1,
                           icon_links= [
                              {
                                 # Label for this link
                                 "name": "GitHub",
                                 # URL where the link will redirect
                                 "url": "https://github.com/BU-SMBL/Tim-Josephson",  # required
                                 # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
                                 "icon": "fa-brands fa-square-github",
                                 # The type of image to be used (see below for details)
                                 "type": "fontawesome",
                              },],
                           pygments_light_style=pygments_light_style, 
                           pygments_dark_style=pygments_dark_style
                        )


html_context = {
   "default_mode": "light"
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'mymesh': ('https://bu-smbl.github.io/mymesh/', None)
}

# Plotting options
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ['png']
plot_pre_code = '''
'''

graphviz_dot = r"dot"
copybutton_prompt_text = ">>> "

bibtex_bibfiles = ['references.bib']
