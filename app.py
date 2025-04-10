from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import psycopg2
from typing import List, Dict, Tuple, Optional
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Configuración de PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'database': 'rutas_ecuador',
    'user': 'postgres',
    'password': 'root',
    'port': '5432'
}

class Graph:
    def __init__(self):
        self.vertices = {}  # {id: (nombre, lat, lon)}
        self.edges = {}     # {id: [vecinos]}
    
    def add_vertex(self, city_id: int, name: str, lat: float, lon: float):
        self.vertices[city_id] = (name, lat, lon)
        self.edges[city_id] = []
    
    def add_edge(self, city1: int, city2: int):
        if city2 not in self.edges[city1]:
            self.edges[city1].append(city2)
        if city1 not in self.edges[city2]:
            self.edges[city2].append(city1)
    
    def haversine_distance(self, city1_id: int, city2_id: int) -> float:
        """Calcula distancia usando la fórmula de Haversine"""
        lat1, lon1 = self.vertices[city1_id][1], self.vertices[city1_id][2]
        lat2, lon2 = self.vertices[city2_id][1], self.vertices[city2_id][2]
        
        R = 6371  # Radio de la Tierra en km
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dLon / 2) * math.sin(dLon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def total_cost(self, city1_id: int, city2_id: int) -> float:
        return self.haversine_distance(city1_id, city2_id)

def create_db_connection():
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        port=DB_CONFIG['port']
    )

UMBRAL_CONEXION = 100

def initialize_graph() -> Graph:
    graph = Graph()
    conn = create_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Cargar ciudades
    cursor.execute("SELECT id, nombre, latitud, longitud FROM ciudades")
    cities = cursor.fetchall()
    for city in cities:
        graph.add_vertex(city['id'], city['nombre'], city['latitud'], city['longitud'])
    
    # Crear conexiones (aristas) entre ciudades cercanas
    city_ids = [city['id'] for city in cities]
    for i, city1 in enumerate(city_ids):
        for city2 in city_ids[i+1:]:
            if graph.haversine_distance(city1, city2) < UMBRAL_CONEXION:
                graph.add_edge(city1, city2)
                print(f"Conexión creada: {city1} - {city2}")  # Debug
    
    conn.close()
    return graph

# Algoritmos de búsqueda de ruta
def dijkstra(graph: Graph, start: int, end: int) -> Tuple[List[int], float]:
    """Implementación mejorada del algoritmo de Dijkstra"""
    import heapq
    
    queue = []
    heapq.heappush(queue, (0, start))
    costs = {start: 0}
    predecessors = {start: None}
    visited = set()
    
    while queue:
        current_cost, current_node = heapq.heappop(queue)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == end:
            break
        
        for neighbor in graph.edges[current_node]:
            if neighbor not in visited:
                new_cost = current_cost + graph.total_cost(current_node, neighbor)
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    predecessors[neighbor] = current_node
                    heapq.heappush(queue, (new_cost, neighbor))
    
    # Reconstruir el camino solo si se encontró una ruta
    if end not in predecessors:
        return [end], float('inf')
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors.get(current, None)
    path.reverse()
    
    return path, costs.get(end, float('inf'))

def a_star(graph: Graph, start: int, end: int) -> Tuple[List[int], float]:
    """Implementación del algoritmo A*"""
    import heapq
    
    def heuristic(node):
        # Usamos distancia directa como heurística
        return graph.haversine_distance(node, end)
    
    queue = []
    heapq.heappush(queue, (0 + heuristic(start), start))
    costs = {start: 0}
    predecessors = {start: None}
    
    while queue:
        _, current_node = heapq.heappop(queue)
        
        if current_node == end:
            break
        
        for neighbor in graph.edges[current_node]:
            new_cost = costs[current_node] + graph.total_cost(current_node, neighbor)
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (new_cost + heuristic(neighbor), neighbor))
    
    # Reconstruir el camino
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors.get(current, None)
    path.reverse()
    
    return path, costs.get(end, float('inf'))

def greedy_search(graph: Graph, start: int, end: int) -> Tuple[List[int], float]:
    """Implementación de Búsqueda Voraz (Primero el mejor)"""
    import heapq
    
    def heuristic(node):
        return graph.haversine_distance(node, end)
    
    queue = []
    heapq.heappush(queue, (heuristic(start), start))
    predecessors = {start: None}
    visited = set()
    total_cost = 0
    
    while queue:
        _, current_node = heapq.heappop(queue)
        visited.add(current_node)
        
        if current_node == end:
            break
        
        for neighbor in graph.edges[current_node]:
            if neighbor not in visited:
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (heuristic(neighbor), neighbor))
    
    # Reconstruir el camino y calcular costo total
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors.get(current, None)
    path.reverse()
    
    # Calcular costo total del camino
    cost = 0
    for i in range(len(path)-1):
        cost += graph.total_cost(path[i], path[i+1])
    
    return path, cost

# Rutas de la API
@app.route('/api/cities', methods=['GET', 'POST'])
def handle_cities():
    """Manejador para obtener y agregar ciudades"""
    conn = create_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if request.method == 'GET':
        cursor.execute("SELECT id, nombre, latitud, longitud FROM ciudades")
        cities = cursor.fetchall()

        # 👇 Asegurar que latitud y longitud sean floats
        for city in cities:
            city['latitud'] = float(city['latitud'])
            city['longitud'] = float(city['longitud'])

        conn.close()
        return jsonify(cities)
    
    elif request.method == 'POST':
        data = request.get_json()
        nombre = data.get('nombre')
        latitud = data.get('latitud')
        longitud = data.get('longitud')
        
        cursor.execute(
            "INSERT INTO ciudades (nombre, latitud, longitud) VALUES (%s, %s, %s) RETURNING id",
            (nombre, latitud, longitud)
        )
        new_id = cursor.fetchone()['id']
        conn.commit()
        conn.close()
        
        return jsonify({
            'id': new_id,
            'nombre': nombre,
            'latitud': latitud,
            'longitud': longitud
        }), 201

@app.route('/api/cities/<int:city_id>', methods=['PUT', 'DELETE'])
def handle_city(city_id):
    """Actualiza o elimina una ciudad existente"""
    conn = create_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if request.method == 'PUT':
        data = request.get_json()
        nombre = data.get('nombre')
        latitud = data.get('latitud')
        longitud = data.get('longitud')
        
        try:
            cursor.execute(
                "UPDATE ciudades SET nombre = %s, latitud = %s, longitud = %s WHERE id = %s RETURNING id, nombre, latitud, longitud",
                (nombre, latitud, longitud, city_id)
            )
            updated_city = cursor.fetchone()
            conn.commit()
            
            if updated_city:
                # Convertir a float para asegurar el formato correcto en la respuesta
                updated_city['latitud'] = float(updated_city['latitud'])
                updated_city['longitud'] = float(updated_city['longitud'])
                return jsonify(updated_city)
            else:
                return jsonify({'error': f'Ciudad con ID {city_id} no encontrada'}), 404
                
        except Exception as e:
            conn.rollback()
            return jsonify({'error': f'Error al actualizar la ciudad: {str(e)}'}), 500
        finally:
            conn.close()
    
    elif request.method == 'DELETE':
        try:
            # Verificar si la ciudad existe antes de eliminarla
            cursor.execute("SELECT id FROM ciudades WHERE id = %s", (city_id,))
            if cursor.fetchone() is None:
                return jsonify({'error': f'Ciudad con ID {city_id} no encontrada'}), 404
            
            # Eliminar la ciudad
            cursor.execute("DELETE FROM ciudades WHERE id = %s", (city_id,))
            conn.commit()
            return jsonify({'message': f'Ciudad con ID {city_id} eliminada correctamente'}), 200
        
        except Exception as e:
            conn.rollback()
            return jsonify({'error': f'Error al eliminar la ciudad: {str(e)}'}), 500
        finally:
            conn.close()

@app.route('/api/route', methods=['GET'])
def calculate_route():
    """Calcula la ruta entre dos ciudades usando el algoritmo especificado"""
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    algorithm = request.args.get('algorithm', 'dijkstra')
    
    graph = initialize_graph()
    
    if algorithm == 'dijkstra':
        path, cost = dijkstra(graph, start, end)
    elif algorithm == 'astar':
        path, cost = a_star(graph, start, end)
    elif algorithm == 'greedy':
        path, cost = greedy_search(graph, start, end)
    else:
        return jsonify({'error': 'Algoritmo no válido'}), 400
    
    # Obtener detalles de las ciudades en el camino
    cities_in_path = []
    for city_id in path:
        name, lat, lon = graph.vertices[city_id]
        cities_in_path.append({
            'id': city_id,
            'nombre': name,
            'latitud': lat,
            'longitud': lon
        })
    
    return jsonify({
        'path': path,
        'cost': cost,
        'cities': cities_in_path
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050)