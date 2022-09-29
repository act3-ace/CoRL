.PHONY: lockall
lockall: requirements.lock.txt mkdocs-requirements.lock.txt setup.lock.txt

.PHONY: relockall
relockall: remove_lock_file lockall

requirements.lock.txt: requirements.txt
	pip install --dry-run --ignore-installed -r requirements.txt > requirements.lock.txt
	python3 scripts/piplock_tool.py requirements.lock.txt

mkdocs-requirements.lock.txt: mkdocs-requirements.txt 
	pip install --dry-run --ignore-installed -r mkdocs-requirements.txt > mkdocs-requirements.lock.txt
	python3 scripts/piplock_tool.py mkdocs-requirements.lock.txt

setup.lock.txt: setup.py 
	pip install --dry-run --ignore-installed -e . > setup.lock.txt
	python3 scripts/piplock_tool.py setup.lock.txt

.PHONY: remove_lock_file
remove_lock_file:
	rm requirements.lock.txt mkdocs-requirements.lock.txt setup.lock.txt -f