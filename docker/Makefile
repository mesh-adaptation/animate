all: build

build:
	docker build -f Dockerfile.firedrake -t jwallwork/firedrake-parmmg --no-cache .

pull:
	docker pull jwallwork/firedrake-parmmg

run:
	docker run --rm -it -v ${HOME}:${HOME} jwallwork/firedrake-parmmg
