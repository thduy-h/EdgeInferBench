{
  description = "EdgeInferBench CPU wheel development shell (no CUDA required)";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      eachSystem = nixpkgs.lib.genAttrs systems;
    in {
      devShells = eachSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };
          runtimeLibraries = with pkgs; [ stdenv.cc.cc.lib zlib glib libGL ];
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [ python311 uv git ];
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = "${pkgs.python311}/bin/python3";
            shellHook = ''
              echo "EdgeInferBench: run uv sync --extra dev (optionally --extra openvino)"
            '';
          };
        });
    };
}
