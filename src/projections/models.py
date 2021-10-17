class Records:
    def __init__(self, shapes_by_country: dict, country=None):
        self.country = country
        self.shapes_by_country = shapes_by_country
        self.total_shapes = sum(len(s) for s in shapes_by_country.values())
        if self.country is not None: 
            assert self.country in shapes_by_country, f'Country {self.country} not found!'
        
    def __iter__(self):
        if self.country is not None:            
            yield from self.shapes_by_country[self.country]
        else:
            for shapes in self.shapes_by_country.values():
                yield from shapes
            
    def __len__(self):
        if self.country is not None:
            return len(self.shapes_by_country[self.country])
        else:
            return self.total_shapes