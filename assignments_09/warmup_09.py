#======= Azure Authentiction =====

#Question 1 
# when you run a Python script locally that uses DefaultAzureCredential, what does it rely on to authenticate? What command must you have run first, and how does DefaultAzureCredential know to use it?
"""  The DefaultAzureCredential rely on the az login session , where you run 'az login'. It then tries multiple sequence and uses the first one that succeeds. Locally the az login works so the DefaultAzureCredential automatically
pick up the sequence without needing a credential code.
"""

#Question 2 
#why can't a deployed pipeline (running on an Azure VM or container) use az login for authentication? What does it use instead, and why does the same Python code work without changes?
""" The deploy pipeline (on an Azure VM or container) cant use az login because it requires a human log in. Thats where managed identity comes into play , which is automatically detected by DefaultAzureCredential. 
The the python script works as well in the deply pipeline because then the DefaultAzureCredential tried again iwith multi sequence till one succeeds
 """

#Question 3 
#You run a script that creates a DefaultAzureCredential and immediately gets an AuthenticationError. In a comment block, describe the two most likely causes and how you would diagnose each.
""" Scenario 1 : would be if the az login sessions has expired or was never running again iwth the command 'az login'
Scenario 2: The credentials that DefaultAzureCredentails chceks it dooesnt exist or hasnt been set up. Evene veriifying the VM has a managed identity."""

#========== Blob Storage ==========
#In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.)
"""The Azure Blob Storage has 3 levels: 
1. Storage Account : Each account has a unique name and URl form 
2. Container: grouping of blob within the stprage account - you can have many container in oe account. 
It can be one container per pipeline or project
3.Blob: individual file, storred by name , names included slash 
"""

#Question 2 
#For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.

    #A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
    #1. Blob since you will retrieve the raw files.

   # Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
   #2. Azure SQL database , simce it a thounsand data to be anaylze 

    #A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.
    #Blob Storage since its raw files , irregualr data that is being passed by pipeline all at once.

#Question 3 
def list_container(container_client):
    """ Print the name and size(in bytes) of every blob in the container , one per line
    """
    for blob in container_client.list.blobs():
        print(f"{blob.name} {blob.size}")
    else:
        return 0


#Question 4 : 
def upload_text(container_client, blob_name, text):
    """Encode a Python script as UTF-8 and upload it as a blob, overwritting any existing blob with the same name"""

    container_client.upload_blob(blob_name, text.encode("utf-8"), overwrite=True)