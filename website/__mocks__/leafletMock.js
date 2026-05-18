module.exports = {
  map: jest.fn(),
  tileLayer: jest.fn(),
  imageOverlay: jest.fn(),
  latLng: jest.fn().mockReturnValue({ lat: 0, lng: 0 }),
  latLngBounds: jest.fn().mockReturnValue({
    isValid: () => true,
    toBBoxString: () => '',
    getNorthEast: jest.fn(),
    getSouthWest: jest.fn(),
  }),
  Icon: { Default: { mergeOptions: jest.fn() } },
};
