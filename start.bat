@echo off
set ip_address_string="IPv4"
echo Network Connection Test
for /f "usebackq tokens=2 delims=:" %%f in (`ipconfig ^| findstr /c:%ip_address_string%`) do (
	set IP_MAIN=%%f
	goto :docker
)
:docker
set IP_MAIN=%IP_MAIN: =%

docker-compose up -d