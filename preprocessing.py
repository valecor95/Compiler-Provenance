import json

def preprocess(file):
    f = open(file)
    lines = f.readlines()
    f.close()
    dataset = list(map(lambda x: x, lines))									# scarta i doppioni REPORT
    json_D = []

    # Save mnemonics + MEMORY CALL + REGISTER
    for l in dataset:
    	function = json.loads(l)
    	for i in range(len(function["instructions"])):
    		instruction0 = list()
    		instruction0.append(function["instructions"][i].split(" ")[0])
    		instruction = function["instructions"][i].split(" ")
    		# Memory call
    		if "dword" in instruction or "word" in instruction or "qword" in instruction or "byte" in instruction:
    			instruction0.append("memory")
    		#Data Registers
    		if "rax" in instruction or "eax" in instruction or "al" in instruction:
    			instruction0.append("rax")
    		if "rbx" in instruction or "ebx" in instruction or "bl" in instruction:
    			instruction0.append("rbx")
    		if "rcx" in instruction or "ecx" in instruction or "cl" in instruction:
    			instruction0.append("rcx")
    		if "rdx" in instruction or "edx" in instruction or "dl" in instruction:
    			instruction0.append("rdx")
    		# Pointer Registers
    		if "rbp" in instruction or "ebp" in instruction:
    			instruction0.append("rbp")
    		if "rsp" in instruction or "esp" in instruction:
    			instruction0.append("rsp")
    		# Index Registers
    		if "rdi" in instruction or "edi" in instruction:
    			instruction0.append("rdi")
    		if "rsi" in instruction or "esi" in instruction:
    			instruction0.append("rsi")
    		function["instructions"][i] = ' '.join(instruction0)
    	# Make string for vectorizer
    	function["instructions"] = ' '.join(function["instructions"])
    	json_D.append(function)
    return json_D
