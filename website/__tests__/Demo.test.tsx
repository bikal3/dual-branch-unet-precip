import { render, screen, fireEvent } from '@testing-library/react';
import Demo from '@/components/Demo';

// DemoMap is dynamically imported — mock next/dynamic so tests don't need Leaflet
jest.mock('next/dynamic', () => () => {
  const MockMap = ({ date, opacity }: { date: string; opacity: number }) => (
    <div data-testid="demo-map" data-date={date} data-opacity={opacity} />
  );
  MockMap.displayName = 'MockMap';
  return MockMap;
});

const BASE_PATH = '';

describe('Demo', () => {
  it('renders a day button for each sample date', () => {
    render(<Demo basePath={BASE_PATH} />);
    expect(screen.getByRole('button', { name: /jan 15/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /mar 10/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /jul 20/i })).toBeInTheDocument();
  });

  it('renders the opacity slider', () => {
    render(<Demo basePath={BASE_PATH} />);
    expect(screen.getByRole('slider')).toBeInTheDocument();
  });

  it('passes the first date to the map by default', () => {
    render(<Demo basePath={BASE_PATH} />);
    expect(screen.getByTestId('demo-map')).toHaveAttribute('data-date', '2020-01-15');
  });

  it('changes the selected date when a day button is clicked', () => {
    render(<Demo basePath={BASE_PATH} />);
    fireEvent.click(screen.getByRole('button', { name: /mar 10/i }));
    expect(screen.getByTestId('demo-map')).toHaveAttribute('data-date', '2020-03-10');
  });

  it('has the correct section id', () => {
    const { container } = render(<Demo basePath={BASE_PATH} />);
    expect(container.querySelector('#demo')).not.toBeNull();
  });
});
