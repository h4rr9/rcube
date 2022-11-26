clean:
	cargo clean
	rm -rf *~ dist *.egg-info build target

build-pyo3:
	maturin build -i python3 --release --features python

develop-pyo3:
	maturin develop --release --features python

test:
	cargo test 
	nose2 -vv -t ./train -s ./train --log-capture

bench:
	cargo bench
