from __future__ import annotations

from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

# Compiler settings
Options.fast_fail = False
Options.annotate  = False
directives        = {
	"boundscheck":      True,
	"wraparound":       True,
	"initializedcheck": True,
	"nonecheck":        True,
}

inc = [np.get_include(), "."]

files = {
	"deuceCard":     ["strategos_deuces/card.pyx"],
	"deuceDeck":     ["strategos_deuces/deck.pyx"],
	"deuceLookup":   ["strategos_deuces/lookup.pyx"],
	"deuceEval":     ["strategos_deuces/evaluator.pyx"],
	"CONSTS":        ["strategos_tools/core/CONSTS.pyx"],
	"containers":    ["strategos_tools/core/containers.pyx"],
	"card_ops":      ["strategos_tools/env/card_ops.pyx"],
	"player_ops":    ["strategos_tools/env/player_ops.pyx"],
	"event_ops":     ["strategos_tools/env/event_ops.pyx"],
	"gamenode_ops":  ["strategos_tools/env/gamenode_ops.pyx"],
	"infoset_ops":   ["strategos_tools/env/infoset_ops.pyx"],
	"actionset_ops": ["strategos_tools/env/actionset_ops.pyx"],
	"utilfuncs":     ["strategos_tools/utils/funcs.pyx"],
	"data_structs":  ["strategos_tools/utils/data_structs.pyx"],
	"data_ops":      ["strategos_tools/utils/data_ops.pyx"],
	"EstimatorOps":  ["strategos_tools/AIOps/EstimatorOps.pyx"],
	"CollectionOps": ["strategos_tools/CFR/CollectionOps.pyx"],
	# NOTE: EvalOps intentionally excluded for now.
}

extensions = [
	# strategos_deuces
	Extension( "strategos_deuces.card",              files['deuceCard'],     include_dirs=inc ),
	Extension( "strategos_deuces.deck",              files['deuceDeck'],     include_dirs=inc ),
	Extension( "strategos_deuces.lookup",            files['deuceLookup'],   include_dirs=inc ),
	Extension( "strategos_deuces.evaluator",         files['deuceEval'],     include_dirs=inc ),

	# strategos_tools.core
	Extension( "strategos_tools.core.CONSTS",        files['CONSTS'],        include_dirs=inc ),
	Extension( "strategos_tools.core.containers",    files['containers'],    include_dirs=inc ),

	# strategos_tools.env
	Extension( "strategos_tools.env.card_ops",       files['card_ops'],      include_dirs=inc ),
	Extension( "strategos_tools.env.player_ops",     files['player_ops'],    include_dirs=inc ),
	Extension( "strategos_tools.env.event_ops",      files['event_ops'],     include_dirs=inc ),
	Extension( "strategos_tools.env.gamenode_ops",   files['gamenode_ops'],  include_dirs=inc ),
	Extension( "strategos_tools.env.infoset_ops",    files['infoset_ops'],   include_dirs=inc ),
	Extension( "strategos_tools.env.actionset_ops",  files['actionset_ops'], include_dirs=inc ),

	# strategos_tools.utils
	Extension( "strategos_tools.utils.funcs",        files['utilfuncs'],     include_dirs=inc ),
	Extension( "strategos_tools.utils.data_structs", files['data_structs'],  include_dirs=inc ),
	Extension( "strategos_tools.utils.data_ops",     files['data_ops'],      include_dirs=inc ),

	# strategos_tools.AIOps
	Extension( "strategos_tools.AIOps.EstimatorOps", files['EstimatorOps'],  include_dirs=inc ),

	# strategos_tools.CFR
	Extension( "strategos_tools.CFR.CollectionOps",  files['CollectionOps'], include_dirs=inc ),
]

setup(
	name="strategos",
	version="0.0.0",
	description="Strategos Cython extensions and tooling",
	python_requires=">=3.12",
	zip_safe=False,

	# Install only the actual packages you care about.
	# (EvalOps intentionally excluded for now; no need to remove its __init__.py.)
	packages=find_packages(
		include=["strategos_deuces*", "strategos_tools*"],
		exclude=[
			"strategos_tools.EvalOps",
			"strategos_tools.EvalOps.*",
		],
	),

	ext_modules=cythonize(
		extensions,
		language_level="3",
		compiler_directives=directives,
	),

	# Runtime deps. (Build deps like Cython/numpy should be declared in a root
	# pyproject.toml; setup.py alone is not reliable for isolated builds.)
	install_requires=["numpy"],
)