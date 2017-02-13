.PHONY: all build-kabukinai clean

all: build-kabukinai

build-kabukinai: build/ build/Makefile
	make -C $<

build/:
	mkdir -p $@

build/Makefile: build/
	(cd $< ; cmake ..)

clean:
	rm -r build/
