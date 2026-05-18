const React = require('react');

module.exports = {
  MapContainer: ({ children }) => React.createElement('div', { 'data-testid': 'map' }, children),
  TileLayer: () => null,
  ImageOverlay: () => null,
};
