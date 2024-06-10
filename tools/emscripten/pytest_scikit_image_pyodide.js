// A JavaScript file to run the scikit-image test suite using Pyodide
// This file is used by the GitHub Actions workflow to run the tests
// against the Pyodide build of scikit-image defined in emscripten.yml.

// The contents of this file are attributed to the scikit-learn developers,
// who have a similar file in their repository:
// https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/azure/pytest-pyodide.js


const { opendir } = require('node:fs/promises');
const { loadPyodide } = require("pyodide");

async function main() {
    let exit_code = 0;
    try {
        global.pyodide = await loadPyodide();
        let pyodide = global.pyodide;
        const FS = pyodide.FS;
        const NODEFS = FS.filesystems.NODEFS;

        let mountDir = "/mnt";
        pyodide.FS.mkdir(mountDir);
        pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: "." }, mountDir);

        await pyodide.loadPackage(["micropip"]);
        await pyodide.runPythonAsync(`
            import glob
            import micropip

            wheels = glob.glob("/mnt/dist/*.whl")
            wheels = [f'emfs://{wheel}' for wheel in wheels]
            print(f"Installing wheels: {wheels}")
            await micropip.install(wheels);

            pkg_list = micropip.list()
            print(pkg_list)
        `);


        await pyodide.runPythonAsync("import micropip; micropip.install('pytest')");
        await pyodide.runPythonAsync("import micropip; micropip.install('matplotlib')");
        let pytest = pyodide.pyimport("pytest");
        let args = process.argv.slice(2);
        console.log('pytest args:', args);
        exit_code = pytest.main(pyodide.toPy(args));
    } catch (e) {
        console.error(e);
        // Arbitrary exit code here. I have seen this code reached instead of a
        // Pyodide fatal error sometimes
        exit_code = 66;

    } finally {
        process.exit(exit_code);
    }
}

main();
