development = "local"
if development == "local":
    runHost = "192.168.0.146"
    fullPath = 'C:\\python\\chronicare'
    runDebug = True
elif development == "doscom":
    runHost = "172.16.0.41"
    fullPath = '/home/flbadrul/chronicare'
    runDebug = False
runPort = 5000