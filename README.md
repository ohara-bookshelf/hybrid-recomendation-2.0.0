
# The model interface API is run on docker.\n
## To run the model all you need to do is:
## 1. Clone the source code
## 2. Navigate the interface forder 
## 3. Run the docker by the following:
### 3-a. Build the docker image by this command:
`docker build -t hybrid-recommendation-image .`
### 3-b. Create a container for the image and run it:
`docker run -d -p 8001:80 --name hybrid-recommendation-container hybrid-recommendation-image`


# How to use the API:

## 1. Locate the API on the URL:
`http://localhost/docs`
## 2.Click recommend
## 3. Click on the "Try it out" button, then on the json file enter the book name on the string text and number of books you want to get recommended on the count field. Finally click "execute" button. The numbers shown are the ISBN of the recommended books.


## Engooooy it!!!