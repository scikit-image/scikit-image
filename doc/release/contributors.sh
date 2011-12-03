git log $1..HEAD --format='* %aN' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u

