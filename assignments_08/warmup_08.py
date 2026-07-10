#=========== Cloud Concept =============

#==== Cloud Concept Question 1 :
#What is the core economic model of cloud computing, and how does it differ from owning your own servers?
#Core economic model is a cloud that uses a pay-as-you-go where you paid for resources you actualy use, shifting away from capital expenditure.
#Its different from owning services since there is no hardware costs , no infrastructure management, you can easily scale resources up or down , and there is lwoer risk.

#==== Cloud Concept Questioon 2 :
#Vertical scaling : adding more power to a single machine (more CPU,RAM, GPU)
#Horizontal Scaling: Adding more machines to distribute evenly the load

#1. Viral product launch: Horizontal scaling applies to this since you can distribute traffic across many servers.
#single machine cany handle the sudden load surge.
#2. Data scientist slow model training:
#Data scientist model traning: Vertical scaling since they need faster GPU and more RAM
#3.Data pipeline: Horizontal scaling applies since it 10 to 10,000 files and its splits to several machines.

#==== Cloud Concept Question 3 
#Gmail : SaaS ,ready to use app with no management needed.

#Azure Virtual Machines: IaaS , manage everything above the infrastructure

#Azure App Service: PaaS , Azure handles the infrastructure and operating system, while you manage the code.

#AWS S3 (Simple Storage Service): IaaS, manage what is in store and the data.

#GitHub Codespaces: PaaS; Github handles the infrastructure , you handle the code.

#Snowflake: SaaS: You manage all from the data but Snowflakes handles the infrastructure.

#IaaS (Infrastructure as a Service): You are renting the computing resources , managing the OS,application, data.
#Providers manage the server, storage, and network.

#PaaS(Platform as a Service): Rent a platform fo building and deploying apps.
#Provider manages everything else 

#SaaS(Software as a Service): fully manage , ready to use application over the internet.
#Providers manage evrything else.

#==== Cloud Concepts Question 4:
#What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

#manage data platform are specialize software that is built on tp of the cloud infrastructure , its used data warehousing and analytics.
#the difference with cloud is that the cloud is general infrastructure.

#What you gain is less infrastructure , computing and storage are separated,better performance. However this is specialize therefor you give up flexibility . cost control, and multi purpose in using more then one machine.

#===== Cloud Concepting Question 5:
#Cloud isnt ideal if your dataset fit comfortably on a single machine annd you dont have  a massive demand , so local processing would be faster and cheaper.

#Cloud isnt ideal when if you dont shut the machine properly , you will run up the bill . As even if its pay as you go if you keep it running , the use of it will run as well which leads to a hefty bill.

#=============== Azure Basics ============
#===== Azure Basic Question 1 
#Subscription is your billing , contains all your resoruces and services , the subscriptioon is tied to YOUR account.
#While resources group is a container for organizer related courses, shared with CTD.  
#Azure subscription is mine alone , while resource grop is CTD.

#==== Azure Basic Question 2 
#What that means in practice is that you cant save scripts or files in cloud shell and expect them to be there, 
#when you log in. 
#Persistent Storage in the course setup //odalissctd2026sa.file.core.windows.net/home on /usr/csuser/clouddrive

#===== Azure Basic Question 3 
#What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?
#Your ssh private key is the password , while your public key is like the lock to match the key. 
#Private key is used to connect servers while public key is uploaded to the remote servers.
#Private key stays on the local machine while public key goes to the remote system.
#It is safer this way as its one-way encryption where the public key cant decrypt data only pivate key.
#The authentication that comes with using it. It is more secure then using a password.

#==== Azure Basic Question 4 
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "mail#lillyperez9@icloud.com",
    "type": "user"
  }
}

#then added --output table 
#EnvironmentName    HomeTenantId                          IsDefault    Name                       State    TenantId
-----------------  ------------------------------------  -----------  -------------------------  -------  ------------------------------------
#AzureCloud         0f040ddd-301f-4665-8677-7b21f129d605  True         CTD Nonprofit Sponsorship  Enabled  0f040ddd-301f-4665-8677-7b21f129d605

#Instead of it being in a code format , it gave out a table of the information. 

#======== Part 1: Portal walkthrough and Part 2: Cost Analysis in project_08.md

