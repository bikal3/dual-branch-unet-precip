import type { Config } from 'jest';

const config: Config = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
    '\\.(css|less|scss|sass)$': '<rootDir>/__mocks__/styleMock.js',
    '^leaflet$': '<rootDir>/__mocks__/leafletMock.js',
    '^react-leaflet$': '<rootDir>/__mocks__/reactLeafletMock.js',
    '\\.(png|jpg|jpeg|gif|svg|ico|webp)$': '<rootDir>/__mocks__/fileMock.js',
  },
  transform: {
    '^.+\\.(ts|tsx)$': ['babel-jest', { presets: ['next/babel'] }],
  },
  transformIgnorePatterns: [
    '/node_modules/(?!(react-leaflet|@react-leaflet)/).*',
  ],
};

export default config;
