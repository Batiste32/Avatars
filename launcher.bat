@echo
set /p user_parameter=Name of the character to load :
call .venv/Scripts/activate
python main.py %user_parameter%
pause
