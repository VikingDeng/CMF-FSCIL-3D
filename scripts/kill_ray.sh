ps -def | grep ray | cut -c 9-15 | xargs kill -9