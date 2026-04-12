"use client";

import { useMemo, useState } from "react";
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

type MapPickerProps = {
  lat: number;
  lon: number;
  onChange: (lat: number, lon: number) => void;
};

type SearchResult = {
  display_name: string;
  lat: string;
  lon: string;
};

// Fix default marker icons when bundling with Next.js.
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function RecenterMap({ lat, lon }: { lat: number; lon: number }) {
  const map = useMap();
  map.setView([lat, lon], map.getZoom(), { animate: true });
  return null;
}

function ClickHandler({ onChange }: { onChange: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(event) {
      onChange(event.latlng.lat, event.latlng.lng);
    },
  });
  return null;
}

export default function MapPicker({ lat, lon, onChange }: MapPickerProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);

  const markerPosition = useMemo(() => [lat, lon] as [number, number], [lat, lon]);

  async function handleSearch() {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setSearching(true);
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(query.trim())}`;
      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
      });
      const data = (await response.json()) as SearchResult[];
      setResults(Array.isArray(data) ? data : []);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-dim">Select Location on Map</p>
      <div className="flex gap-2">
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search location"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleSearch();
            }
          }}
        />
        <button type="button" className="button-primary px-3 py-2" onClick={handleSearch}>
          {searching ? "..." : "Go"}
        </button>
      </div>

      {results.length > 0 ? (
        <div className="max-h-32 overflow-auto rounded-xl border border-borderSoft bg-slate-900/80">
          {results.map((item) => (
            <button
              key={`${item.lat}-${item.lon}-${item.display_name}`}
              type="button"
              className="block w-full border-b border-borderSoft px-3 py-2 text-left text-xs text-dim hover:bg-white/5"
              onClick={() => {
                onChange(Number(item.lat), Number(item.lon));
                setResults([]);
                setQuery(item.display_name);
              }}
            >
              {item.display_name}
            </button>
          ))}
        </div>
      ) : null}

      <div className="overflow-hidden rounded-xl border border-borderSoft">
        <MapContainer center={markerPosition} zoom={10} scrollWheelZoom className="h-[220px] w-full">
          <TileLayer
            attribution="Tiles &copy; Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
          <Marker position={markerPosition} />
          <ClickHandler onChange={onChange} />
          <RecenterMap lat={lat} lon={lon} />
        </MapContainer>
      </div>
      <p className="text-xs text-dim">{lat.toFixed(6)}, {lon.toFixed(6)}</p>
    </div>
  );
}
