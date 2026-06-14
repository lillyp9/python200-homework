#===== Part 1 : Portal Walkthrough ======

https://drive.google.com/file/d/1QdbxQanRSt2PpsvdRllmjWGVad_ZfVbL/view?usp=sharing


#Part 2 : Cost Analysis 
#Scenario A -- Lightweight Compute: A Standard_B1s VM (1 vCPU, 1 GB RAM) running 8 hours a day, 5 days a week (about 160 hours a month).

#In Scenario A - Lightweight Compute 
#Standard_B1s VM (1 vCPU, 1 GB RAM) running 8 hours a day, 5 days a week (about 160 hours a month).
# The average cost monthly would be $9.60, the yearly cost would be $115.20

#In Scenario B - Heavy analytics workload: 
#A GPU-enabled VM (Standard_NC6s_v3: 6 vCPU, 1 V100 GPU) running 24/7 for the full month (730 hours): 
#Monthly cost: $2,435.28 , while Annual Cost: $29,223.36
#an Azure SQL Database (General Purpose tier, 4 vCores): 
#Monthly cost: $741.16, Annual cost:$8,893.90
#Azure Blob Storage account with 1 TB of data.: 
#Monthly cost: $154.29, Annual Cost: $1,851.48

** Video link for Scenario A **
https://drive.google.com/file/d/1xQ9rBuRbv7uxRc2MO4w0CuAcbxh8Bf9-/view?usp=sharing

** Video link for Scenario B **
https://drive.google.com/file/d/1NgH_4mvaP6Hio69SHjYYfNhKDVFUGRBe/view?usp=sharing


#=== Part3 
https://drive.google.com/file/d/1F6iyKiYZm1qyr2PLugpQiuHxfo-BMfFq/view?usp=sharing

#Cost Anaylsis of Part 3 
#What each scenario costs, and whether the numbers surprised you.
Scenario A (lightweight):       $9.76
Scenario B (GPU VM only):       $14.60

These number does not suprise me , however it had me questions if the INSTANCE , I chosed was wrong. 

Anything interesting you found while exploring the Pricing Calculator beyond the two required scenarios.
What I found interested was of how they are able to keep track of the storage from the log.

What the script printed, and whether the calculated costs matched what you saw in the Pricing Calculator. (They should -- if they don't, note the discrepancy)
The Scenario A did match however Scenario B did not , whoch i think its the INSTANCE i chosed from the task in  hand . 