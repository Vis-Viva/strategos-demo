# LEGACY BUILD PIPELINE. This is not used anymore.

# ----- DEUCES -------------------------------------------------------------------------------------


python strategos_deuces/setup.py build_ext

cp build/lib.linux-x86_64-cpython-312/strategos_deuces/card.cpython-312-x86_64-linux-gnu.so strategos_deuces/card.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/card.cpython-312-x86_64-linux-gnu.so runpod/strategos_deuces/card.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/card.cpython-312-x86_64-linux-gnu.so modeleval/strategos_deuces/card.so

cp build/lib.linux-x86_64-cpython-312/strategos_deuces/deck.cpython-312-x86_64-linux-gnu.so strategos_deuces/deck.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/deck.cpython-312-x86_64-linux-gnu.so runpod/strategos_deuces/deck.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/deck.cpython-312-x86_64-linux-gnu.so modeleval/strategos_deuces/deck.so

cp build/lib.linux-x86_64-cpython-312/strategos_deuces/lookup.cpython-312-x86_64-linux-gnu.so strategos_deuces/lookup.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/lookup.cpython-312-x86_64-linux-gnu.so runpod/strategos_deuces/lookup.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/lookup.cpython-312-x86_64-linux-gnu.so modeleval/strategos_deuces/lookup.so

cp build/lib.linux-x86_64-cpython-312/strategos_deuces/evaluator.cpython-312-x86_64-linux-gnu.so strategos_deuces/evaluator.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/evaluator.cpython-312-x86_64-linux-gnu.so runpod/strategos_deuces/evaluator.so
cp build/lib.linux-x86_64-cpython-312/strategos_deuces/evaluator.cpython-312-x86_64-linux-gnu.so modeleval/strategos_deuces/evaluator.so


# ----- STRATEGOS CORE -----------------------------------------------------------------------------


python strategos_tools/core/setup.py build_ext

cp -r strategos_tools/core/assets runpod/strategos_tools/core
cp -r strategos_tools/core/assets modeleval/strategos_tools/core

cp strategos_tools/core/PYCONSTS.py runpod/strategos_tools/core/PYCONSTS.py
cp strategos_tools/core/PYCONSTS.py modeleval/strategos_tools/core/PYCONSTS.py

cp build/lib.linux-x86_64-cpython-312/strategos_tools/core/CONSTS.cpython-312-x86_64-linux-gnu.so strategos_tools/core/CONSTS.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/core/CONSTS.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/core/CONSTS.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/core/CONSTS.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/core/CONSTS.so

#cp -r strategos_tools/core/suitimgs runpod/
#cp -r strategos_tools/core/suitimgs modeleval/

cp build/lib.linux-x86_64-cpython-312/strategos_tools/core/containers.cpython-312-x86_64-linux-gnu.so strategos_tools/core/containers.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/core/containers.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/core/containers.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/core/containers.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/core/containers.so


# ----- ENV OPS ------------------------------------------------------------------------------------


python strategos_tools/env/setup.py build_ext

cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/player_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/env/player_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/player_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/env/player_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/player_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/env/player_ops.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/card_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/env/card_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/card_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/env/card_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/card_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/env/card_ops.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/event_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/env/event_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/event_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/env/event_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/event_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/env/event_ops.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/gamenode_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/env/gamenode_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/gamenode_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/env/gamenode_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/gamenode_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/env/gamenode_ops.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/infoset_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/env/infoset_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/infoset_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/env/infoset_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/infoset_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/env/infoset_ops.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/actionset_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/env/actionset_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/actionset_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/env/actionset_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/env/actionset_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/env/actionset_ops.so


# ----- UTILS --------------------------------------------------------------------------------------


python strategos_tools/utils/setup.py build_ext

cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/funcs.cpython-312-x86_64-linux-gnu.so strategos_tools/utils/funcs.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/funcs.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/utils/funcs.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/funcs.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/utils/funcs.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/data_structs.cpython-312-x86_64-linux-gnu.so strategos_tools/utils/data_structs.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/data_structs.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/utils/data_structs.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/data_structs.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/utils/data_structs.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/data_ops.cpython-312-x86_64-linux-gnu.so strategos_tools/utils/data_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/data_ops.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/utils/data_ops.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/utils/data_ops.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/utils/data_ops.so


# ----- AI OPS -------------------------------------------------------------------------------------


python strategos_tools/AIOps/setup.py build_ext

cp strategos_tools/AIOps/models.py runpod/strategos_tools/AIOps/models.py
cp strategos_tools/AIOps/models.py modeleval/strategos_tools/AIOps/models.py
cp strategos_tools/AIOps/nn_utils.py runpod/strategos_tools/AIOps/nn_utils.py
cp strategos_tools/AIOps/nn_utils.py modeleval/strategos_tools/AIOps/nn_utils.py
cp strategos_tools/AIOps/training.py runpod/strategos_tools/AIOps/training.py
cp strategos_tools/AIOps/training.py modeleval/strategos_tools/AIOps/training.py

cp build/lib.linux-x86_64-cpython-312/strategos_tools/AIOps/EstimatorOps.cpython-312-x86_64-linux-gnu.so strategos_tools/AIOps/EstimatorOps.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/AIOps/EstimatorOps.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/AIOps/EstimatorOps.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/AIOps/EstimatorOps.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/AIOps/EstimatorOps.so


# ----- CFR OPS ------------------------------------------------------------------------------------


python strategos_tools/CFR/setup.py build_ext

cp build/lib.linux-x86_64-cpython-312/strategos_tools/CFR/CollectionOps.cpython-312-x86_64-linux-gnu.so strategos_tools/CFR/CollectionOps.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/CFR/CollectionOps.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/CFR/CollectionOps.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/CFR/CollectionOps.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/CFR/CollectionOps.so


# ----- EVAL OPS -----------------------------------------------------------------------------------


python strategos_tools/EvalOps/setup.py build_ext

cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/evaltools.cpython-312-x86_64-linux-gnu.so strategos_tools/EvalOps/evaltools.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/evaltools.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/EvalOps/evaltools.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/evaltools.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/EvalOps/evaltools.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/modeleval.cpython-312-x86_64-linux-gnu.so strategos_tools/EvalOps/modeleval.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/modeleval.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/EvalOps/modeleval.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/modeleval.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/EvalOps/modeleval.so

cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/gameplay.cpython-312-x86_64-linux-gnu.so strategos_tools/EvalOps/gameplay.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/gameplay.cpython-312-x86_64-linux-gnu.so runpod/strategos_tools/EvalOps/gameplay.so
cp build/lib.linux-x86_64-cpython-312/strategos_tools/EvalOps/gameplay.cpython-312-x86_64-linux-gnu.so modeleval/strategos_tools/EvalOps/gameplay.so

printf "\n\n==================================================\n"
printf "COPYING COMMAND & CONTROL SCRIPTS\n"
printf "==================================================\n\n"

rm runpod/CFRMonitor.py
rm runpod/CFRCollect.py
rm runpod/CFRTrain.py
cp CFRMonitor.py runpod/
cp CFRCollect.py runpod/
cp CFRTrain.py runpod/
cp -r modeleval runpod/

printf "\n\n==================================================\n"
printf "COMPILE & COPY COMPLETE\n"
printf "==================================================\n\n"