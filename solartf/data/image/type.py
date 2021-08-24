class Image:
    def __init__(self,
                 source_ref,
                 image_shape,
                 class_map=None,
                 instance_nb=0,
                 scene_type=None,
                 weather_condition=None,
                 distractor=None):
        """
        :param source_ref:
        :param image_shape: default is (height, width, depth)
        :param class_map: a dictionary with {label_id: label_text}
        :param instance_nb: number of instance inside the image
        :param scene_type: main idea of an image
        :param weather_condition:
               weather-condition=0 indicates "no weather degradation"
               weather-condition=1 indicates "fog/haze"
               weather-condition=2 indicates "rain"
               weather-condition=3 indicates "snow"
        :param distractor
               distractor=0 indicates "not a distractor"
               distractor=1 indicates "distractor"
        """

        self.source_ref = source_ref
        self.image_shape = image_shape
        self.class_map = class_map
        self.scene_type = scene_type
        self.weather_condition = weather_condition
        self.distractor = distractor
        self.instance_nb = instance_nb

    @property
    def height(self):
        return self.image_shape[0]

    @property
    def width(self):
        return self.image_shape[1]
