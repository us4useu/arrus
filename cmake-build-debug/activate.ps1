
$env:CONAN_OLD_conanenv_PATH=$env:PATH

foreach ($line in Get-Content "/home/pjarosik/src/us4useu/arrus/cmake-build-debug/environment.ps1.env") {
    $var,$value = $line -split '=',2
    $value_expanded = $ExecutionContext.InvokeCommand.ExpandString($value)
    Set-Item env:\$var -Value "$value_expanded"
}

function global:_old_conan_conanenv_prompt {""}
$function:_old_conan_conanenv_prompt = $function:prompt
function global:prompt {
    write-host "(conanenv) " -nonewline; & $function:_old_conan_conanenv_prompt
}