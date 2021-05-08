# PACKED QUERIES


## Installation

1. Install Rust using the guie at https://rustup.rs/

2. Build the binary by running the following command

	```
		cargo build

	```

3. Run the queries using the DIRECT method with a specific data set file using the following command:

	```
		cat query_dir.json5  |  ./target/release/packed-queries   [dataset.file]
	```

4. Run the queries using sketchrefine with a specific data set file using the following command:

        ```
                cat query_sketchrefine.json5  |  ./target/release/packed-queries   [dataset.file]


To use different values of K, the source code need to be changed (line 120) and recompiled.

