import configparser

try:
    config = configparser.ConfigParser()
    config.read("../secrets/config.ini")
    passwrd = config["DEFAULT"]["pw"]
    ip = config["DEFAULT"]["ip"]
    login = config["DEFAULT"]["login"]
except KeyError:
    print("credentials needed")
