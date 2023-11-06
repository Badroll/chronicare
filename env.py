development = "local"
runHost = "0.0.0.0"
if development == "local":
    fullPath = 'C:\\python\\chronicare'
    runDebug = True
elif development == "doscom":
    fullPath = '/root/python/chronicare'
    runDebug = False
runPort = 5001