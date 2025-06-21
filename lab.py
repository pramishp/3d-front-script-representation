import os
import requests
from lxml import html

# URL of the webpage
url = 'https://www.zillow.com/homedetails/1000-E-Anderson-St-Orlando-FL-32801/46266378_zpid'

# Make an HTTP request to get the page content
response = requests.get(url)
if response.status_code != 200:
    print("Failed to retrieve the webpage.")
else:
    # Parse the content using lxml
    tree = html.fromstring(response.content)

    # Extract the image URLs using the provided XPath
    image_urls = tree.xpath('//li//img/@src')

    # Create a folder to save images
    if not os.path.exists('images'):
        os.makedirs('images')

    # Download each image
    for i, img_url in enumerate(image_urls):
        # Ensure the URL is absolute
        if not img_url.startswith('http'):
            img_url = 'https:' + img_url

        # Get the image content
        img_data = requests.get(img_url).content

        # Save the image to the local directory
        with open(f'images/image_{i+1}.jpg', 'wb') as file:
            file.write(img_data)
            print(f'Downloaded image_{i+1}.jpg')

print("Download process completed.")
