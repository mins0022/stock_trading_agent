import paramiko

def readTxtFile(fileNm):
    file = open(fileNm+".txt", "r", encoding="UTF-8")
    print(file)
     
    data = []
    while (1):
        line = file.readline()
 
        try:
            escape = line.index('\n')
        except:
            escape = len(line)
        if line:
            data.append(line[0:escape])
        else:
            break
    file.close()
 
    return data
 
def execCommands():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    cli.connect("192.168.0.143", port=8888, username="test", password="test")
 
    commandLines = readTxtFile("./commands/" + "파일명") # 메모장 파일에 적어놨던 명령어 텍스트 읽어옴
    print(commandLines)
 
    stdin, stdout, stderr = cli.exec_command(";".join(commandLines)) # 명령어 실행
    lines = stdout.readlines() # 실행한 명령어에 대한 결과 텍스트
    resultData = ''.join(lines)
 
    print(resultData) # 결과 확인

readTxtFile('frontback/StockTradingAgent/rl_shellscript')
execCommands()
# frontback/StockTradingAgent/rl test.py
# frontback/StockTradingAgent/rl_shellscript