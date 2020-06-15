includes="cpc dataloader emotion_id *.py"
excludes="*.ipynb_checkpoints venv"
ignores="E203, W503, E712"

extensions="@aquirdturtle/collapsible_headings @jupyterlab/toc @wallneradam/output_auto_scroll @wallneradam/run_all_buttons"

check:
	black --check "${includes}"
	flake8 --exclude=${excludes} --ignore=${ignores} --max-line-length=100 "${includes}"
	pylint "${includes}"
clean:
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
deps:
	pip3 install --upgrade pip
	pip3 install -r requirements.txt
	[ -d .git ] && pre-commit install || echo "no git repo to install hooks"
jupyter:
	jupyter labextension install "${extensions}"
	jupyter labextension update "${extensions}"
	jupyter lab build
format:
	black "${includes}"