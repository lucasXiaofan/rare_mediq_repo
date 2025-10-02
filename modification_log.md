**Time**: 2025-09-26T18:23:58.887Z
**Filename**: mediQ/src/expert.py
**Function**: RAREExpert.__init__, RAREExpert.respond, module imports
**Lines**: 1-4, 230-363
**Reason**: Introduce a new RAREExpert class that wraps the RARE pipeline (Generator/search_for_answers) behind the MediQ Expert interface. Adds guarded imports and graceful fallback to implicit abstention when RARE stack is unavailable. Enables running MediQ with a RARE-based expert via --expert_class RAREExpert.