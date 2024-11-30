# SFL
- Need to remove the encryption related code
- Add DP
- Add main server to the fed server with forward & backward propagation
- Update the cloud server being used for ssh and storage

# Configuration
- Change username in server (and/or clients) from `uncleroger` to the corresponding VM's username
- A unique google cloud bucket name. Try first with `hfl-data`.
  - If unsuccessful, try a unique name of your own. And then update the variable *bucket_name* in server and client
- Update the variables `client_private_ips` and `client_public_ips` in server
- Update the variable `client_no` in the client
- Make sure the file `cloud.json` is present in the server and the clients
  - To setup, watch [this](https://www.youtube.com/watch?v=pEbL_TT9cHg) YouTube video (first 4 min)
- Make sure all the VMs have the necessary python packages installed (*scikit, tqdm, tensorflow, etc.*)
- Make sure all the clients have the file named as `test_client.py`
- If you're re-runnning the system, don't forget to run the `server_clean.sh` in server and `clean.sh` in all clients beforehand
