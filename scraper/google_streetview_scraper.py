import requests
import os
import random

# Get api key from environment variable
API_KEY = os.environ.get('GOOGLE_API_KEY')


class GoogleStreetViewScraper:
    def __init__(self, resolution):
        self.resolution = resolution

    def get_images(self, streets, heading_list, save_dir='../data/google_api_images', pitch_min=-10, pitch_max=10):
        for street in streets:
            numbers_list = random.sample(range(1, 50), 15)

            for number in numbers_list:
                # Replace spaces with '+' for URL encoding
                street_query = street.replace(' ', '+') + '+' + str(number)

                for heading in heading_list:
                    # Randomize heading and pitch for different viewpoints
                    pitch = random.randint(pitch_min, pitch_max)

                    # Construct the API URL with varying parameters
                    url = f"https://maps.googleapis.com/maps/api/streetview?size={self.resolution}&location={street_query},+Krakow,+Poland&heading={heading}&pitch={pitch}&key={API_KEY}"

                    # Send request to Google Street View API
                    response = requests.get(url)

                    # Check if the request was successful
                    if response.status_code == 200:
                        # Save the image with a unique name for each viewpoint
                        with open(f'{save_dir}/{street_query}_{heading}.jpg', 'wb') as file:
                            file.write(response.content)

                        print(f"Saved image for {street_query}, viewpoint {heading}")
                    else:
                        print(f"Failed to get image for {street_query}, viewpoint {heading}")


if __name__ == '__main__':
    streets = ['Raclawicka', 'Starowislna', 'Slowackiego']

    os.makedirs('../data/google_api_images', exist_ok=True)

    heading_list = [0, 90, 180, 270]
    scraper = GoogleStreetViewScraper('600x600')
    scraper.get_images(streets, heading_list)
