xcopy %RECIPE_DIR%\bin\windows\t_sne_bhcuda.exe %PREFIX%\Scripts\t_sne_bhcuda.exe* /E
xcopy %RECIPE_DIR%\bin\windows\cudart64_75.dll %PREFIX%\Scripts\cudart64_75.dll* /E
xcopy %RECIPE_DIR%\bin\windows\cudart32_75.dll %PREFIX%\Scripts\cudart32_75.dll* /E

%PYTHON% setup.py install