cat ./resources_list.txt | sed 's/^/-new-tab -url /' | tr '\n' ' ' | xargs firefox & disown
