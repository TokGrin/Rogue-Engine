#include <SFML/Graphics.hpp>
#include <Box2D/Box2D.h>
#include <SFML/Audio.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <map>
#include <queue>       
#include <unordered_set> 

const uint16 PLAYER_CATEGORY = 0x0001;
const uint16 ENEMY_CATEGORY = 0x0002;
const uint16 BULLET_CATEGORY = 0x0004;
const uint16 WALL_CATEGORY = 0x0008;
const uint16 DOOR_CATEGORY = 0x0010;
const uint16 KEY_CATEGORY = 0x0020;
const uint16 HEALTH_CATEGORY = 0x0040;
const uint16 TRAP_CATEGORY = 0x0080;
const uint16 EXIT_CATEGORY = 0x0100;

enum CellType {
    EMPTY = 0,
    WALL = 1,
    PIT = 2,
    PLAYER = 3,
    EXIT = 4,
    ENEMY = 5,
    KEY = 6,
    DOOR = 7,
    TRAP = 8,
    STRONG_ENEMY = 9,
    HEALTH = 10
};

struct Player {
    sf::CircleShape shape;
    sf::ConvexShape directionArc;
    b2Body* body;
    int lives;
    float speed;
    float angle;
    bool hasKey = false;
    int bonusLives = 0;
    int keys = 0;
    float lastShotTime = 0.0f;
    float enemyStartDelayTimer = 3.0f;
    bool enemiesCanMove = false;
};
struct Exit {
    sf::RectangleShape shape;
    b2Body* body;
};

struct Key {
    sf::RectangleShape shape;
    b2Body* body;
    bool collected = false;
};

struct Door {
    sf::RectangleShape shape;
    b2Body* body;
    bool opened = false;
    bool toDestroy = false;  
};

struct HealthPickup {
    sf::CircleShape shape;
    b2Body* body;
    bool active = true;
    float pulseSpeed = 1.0f + (rand() % 100) * 0.01f; 
    float pulseSize = 0.2f; 
    float pulseTime = 0.0f;
};

struct Pit {
    sf::RectangleShape shape;
};

struct Wall {
    sf::RectangleShape shape;
    b2Body* body;
};

struct Heart {
    sf::CircleShape shape;
    bool isBonus; 
};

struct Node {
    int x, y;
    float g, h;
    Node* parent;

    Node(int x, int y, Node* parent = nullptr) : x(x), y(y), g(0), h(0), parent(parent) {}

    float getF() const { return g + h; }
};

std::vector<std::vector<int>> currentLevelMap;
float cellSize = 32.0f;

// Структура врага
struct Enemy {
    sf::CircleShape shape;
    b2Body* body;
    float speed;
    int health;
    bool isStrong = false;
    float stunTimer = 0.0f;
    std::vector<sf::Vector2i> patrolPath;
    size_t currentPatrolPoint = 0;
    float idleTimer = 0.0f;
    std::vector<sf::Vector2i> path;
    float recalculatePathTimer = 0.0f;
    bool toDestroy = false;
    bool isPursuing = false;
    float pursuitTimer = 0.0f;
    float lastSeenPlayerTime = 0.0f;
    float patrolChangeTimer = 0.0f; 
    bool hasSeenPlayer = false;
};

struct Trap {
    sf::ConvexShape shape;
    b2Body* body;
    bool active = true;
    float rotationSpeed = 180.0f; 
};

// Структура пули
struct Bullet {
    sf::CircleShape shape;
    b2Body* body;  
    b2Vec2 direction;
    float distanceTravelled = 0.0f;
    float maxDistance = 160.0f;
    bool toDestroy = false; 
};

std::vector<sf::Vector2i> findPath(const sf::Vector2i& start, const sf::Vector2i& end, const std::vector<Door>& doors);


class RLAgent {
private:
    std::map<std::string, float> qValues;
    float learningRate = 0.1f;
    float discountFactor = 0.9f;
    float explorationRate = 0.3f;
    std::string lastState;
    std::string lastAction;
    float totalReward = 0.0f;
    int episodes = 0;

public:
    struct GameEvent {
        std::string type;
        float reward;
        sf::Vector2f position;
        std::string additionalInfo;
    };

    void update(const GameEvent& event) {
        totalReward += event.reward;

        std::string currentState = encodeState(event);
        std::string actionKey = lastState + "_" + lastAction;

        if (!lastState.empty()) {
            float maxQ = getMaxQValue(currentState);
            qValues[actionKey] += learningRate *
                (event.reward + discountFactor * maxQ - qValues[actionKey]);
        }

        explorationRate = std::max(0.05f, explorationRate * 0.999f);

        lastState = currentState;
        lastAction = chooseAction(currentState);
    }

    void endEpisode() {
        episodes++;
        if (episodes % 10 == 0) {
            saveToFile("rl_agent_state.txt");
        }
        totalReward = 0.0f;
    }

    void saveToFile(const std::string& filename) {
        std::ofstream file(filename);
        for (const auto& [key, value] : qValues) {
            file << key << " " << value << "\n";
        }
        file << "explorationRate " << explorationRate << "\n";
    }

    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (file) {
            std::string key;
            float value;
            while (file >> key >> value) {
                if (key == "explorationRate") {
                    explorationRate = value;
                }
                else {
                    qValues[key] = value;
                }
            }
        }
    }

    std::string encodeState(const GameEvent& event) {
        std::string state = event.type + "_";

        int x = static_cast<int>(event.position.x / cellSize) % 5;
        int y = static_cast<int>(event.position.y / cellSize) % 5;
        state += std::to_string(x) + "_" + std::to_string(y) + "_";

        if (!event.additionalInfo.empty()) {
            state += event.additionalInfo;
        }

        return state;
    }

    std::string chooseAction(const std::string& state) {
        if ((rand() % 100) / 100.0f < explorationRate) {
            return "random_" + std::to_string(rand() % 5);
        }

        std::vector<std::string> actions = {
            "add_enemy_weak", "add_enemy_strong",
            "add_trap", "add_health", "add_key",
            "add_wall", "add_empty"
        };

        float maxQ = -FLT_MAX;
        std::string bestAction = "none";

        for (const auto& action : actions) {
            float q = qValues[state + "_" + action];
            if (q > maxQ || (q == maxQ && rand() % 2 == 0)) {
                maxQ = q;
                bestAction = action;
            }
        }

        return bestAction;
    }

    float getMaxQValue(const std::string& state) {
        float maxQ = -FLT_MAX;
        std::vector<std::string> actions = {
            "add_enemy_weak", "add_enemy_strong",
            "add_trap", "add_health", "add_key"
        };

        for (const auto& action : actions) {
            maxQ = std::max(maxQ, qValues[state + "_" + action]);
        }

        return maxQ;
    }
};

class LevelGenerator {
private:
    std::vector<std::vector<std::vector<int>>> generatedLevels;
    std::mt19937 rng;
    float currentDifficulty = 1.0f;
    float playerSkill = 0.5f;
    std::map<std::string, float> objectWeights;
    float reward = 0.0f;
    const int BASE_ENEMIES = 1;
    const int BASE_STRONG_ENEMIES = 1;
    const int BASE_DOORS = 1;
    const int BASE_KEYS = 1;
    const int BASE_HEALTH = 1;
    RLAgent rlAgent;
    std::vector<RLAgent::GameEvent> gameEvents;
    float totalReward = 0.0f;
    const int maxEnemies = 10;
    const int maxTraps = 5;
    const int maxHealth = 3;
    const int maxKeys = 3;


    const std::vector<std::vector<int>> firstLevel = {
        {1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 8, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,  1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 7, 1, 0, 0, 1, 1, 1, 0, 1, 5, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 8, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 6, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,  0, 1, 1, 0, 0, 1, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    const std::vector<std::vector<int>> finalLevel = {
        {1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
    void generateWalls(std::vector<std::vector<int>>& level, int levelNum) {
        int size = level.size();

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                level[y][x] = WALL;
            }
        }
        generateMaze(level, levelNum);
    }

    void LevelGenerator::generateMaze(std::vector<std::vector<int>>& level, int levelNum) {
        int size = level.size();
        int roomCount = 3 + levelNum; 

        std::vector<sf::Vector2i> roomCenters;
        for (int i = 0; i < roomCount; i++) {
            int roomWidth = 3 + rand() % 4;
            int roomHeight = 3 + rand() % 4;
            int x = 2 + rand() % (size - roomWidth - 4);
            int y = 2 + rand() % (size - roomHeight - 4);
            for (int ry = y + 1; ry < y + roomHeight - 1; ry++) {
                for (int rx = x + 1; rx < x + roomWidth - 1; rx++) {
                    level[ry][rx] = EMPTY;
                }
            }
            roomCenters.emplace_back(x + roomWidth / 2, y + roomHeight / 2);
        }

        for (size_t i = 1; i < roomCenters.size(); i++) {
            connectRoomsWithWalls(level, roomCenters[i - 1], roomCenters[i]);
        }
        for (int y = 2; y < size - 2; y++) {
            for (int x = 2; x < size - 2; x++) {
                if (level[y][x] == EMPTY && rand() % 100 < 60) {
                    level[y][x] = WALL;
                }
            }
        }
    }

    void LevelGenerator::connectRoomsWithWalls(std::vector<std::vector<int>>& level,
        const sf::Vector2i& start,
        const sf::Vector2i& end) {
        int midX = start.x;
        int midY = end.y;
        int stepX = (midX < end.x) ? 1 : -1;
        for (int x = midX; x != end.x; x += stepX) {
            if (level[midY][x] == WALL && (x % 2 == 0)) {
                level[midY][x] = EMPTY;
            }
        }
        int stepY = (start.y < midY) ? 1 : -1;
        for (int y = start.y; y != midY; y += stepY) {
            if (level[y][midX] == WALL && (y % 2 == 0)) { 
                level[y][midX] = EMPTY;
            }
        }
    }

    void connectRooms(std::vector<std::vector<int>>& level, sf::Vector2i start, sf::Vector2i end) {
        int midX = start.x;
        int midY = end.y;
        int stepX = (midX < end.x) ? 1 : -1;
        for (int x = midX; x != end.x; x += stepX) {
            if (level[midY][x] == WALL) level[midY][x] = EMPTY;
        }
        int stepY = (start.y < midY) ? 1 : -1;
        for (int y = start.y; y != midY; y += stepY) {
            if (level[y][midX] == WALL) level[y][midX] = EMPTY;
        }
    }

    bool isLevelPassable(const std::vector<std::vector<int>>& level) {
        sf::Vector2i playerPos, exitPos;
        bool hasPlayer = false, hasExit = false;
        for (size_t y = 0; y < level.size(); y++) {
            for (size_t x = 0; x < level[y].size(); x++) {
                if (level[y][x] == PLAYER) {
                    playerPos = sf::Vector2i(x, y);
                    hasPlayer = true;
                }
                else if (level[y][x] == EXIT) {
                    exitPos = sf::Vector2i(x, y);
                    hasExit = true;
                }
            }
        }

        if (!hasPlayer || !hasExit) return false;
        std::queue<sf::Vector2i> queue;
        std::unordered_set<int> visited;
        auto hash = [](const sf::Vector2i& p) { return p.x * 1000 + p.y; };

        queue.push(playerPos);
        visited.insert(hash(playerPos));

        const std::vector<sf::Vector2i> directions = { {0,1}, {1,0}, {0,-1}, {-1,0} };

        while (!queue.empty()) {
            auto current = queue.front();
            queue.pop();

            if (current == exitPos) return true;

            for (const auto& dir : directions) {
                sf::Vector2i next(current.x + dir.x, current.y + dir.y);
                if (next.x >= 0 && next.y >= 0 &&
                    next.x < (int)level[0].size() && next.y < (int)level.size()) {

                    int cell = level[next.y][next.x];
                    if ((cell == EMPTY || cell == EXIT || cell == KEY ||
                        cell == HEALTH || cell == DOOR || cell == TRAP) &&
                        visited.find(hash(next)) == visited.end()) {

                        visited.insert(hash(next));
                        queue.push(next);
                    }
                }
            }
        }

        return false;
    }

    void applyRLDecisions(std::vector<std::vector<int>>& level) {
        for (int y = 1; y < level.size() - 1; ++y) {
            for (int x = 1; x < level[y].size() - 1; ++x) {
                if (level[y][x] == EMPTY) {
                    RLAgent::GameEvent state;
                    state.type = "cell_empty";
                    state.position = sf::Vector2f(x * cellSize, y * cellSize);
                    int wallsAround = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (level[y + dy][x + dx] == WALL) wallsAround++;
                        }
                    }
                    state.additionalInfo = "walls_" + std::to_string(wallsAround);

                    std::string action = rlAgent.chooseAction(rlAgent.encodeState(state));

                    if (action == "add_enemy_weak" && countEnemies(level) < maxEnemies) {
                        level[y][x] = ENEMY;
                    }
                    else if (action == "add_enemy_strong" && countEnemies(level) < maxEnemies) {
                        level[y][x] = STRONG_ENEMY;
                    }
                    else if (action == "add_trap" && countTraps(level) < maxTraps) {
                        level[y][x] = TRAP;
                    }
                    else if (action == "add_health" && countHealth(level) < maxHealth) {
                        level[y][x] = HEALTH;
                    }
                    else if (action == "add_key" && countKeys(level) < maxKeys) {
                        level[y][x] = KEY;
                    }
                    else if (action == "add_wall") {
                        level[y][x] = WALL;
                    }
                }
            }
        }
    }

    int countEnemies(const std::vector<std::vector<int>>& level) {
        int count = 0;
        for (const auto& row : level) {
            for (int cell : row) {
                if (cell == ENEMY || cell == STRONG_ENEMY) count++;
            }
        }
        return count;
    }

    int countTraps(const std::vector<std::vector<int>>& level) {
        int count = 0;
        for (const auto& row : level) {
            for (int cell : row) {
                if (cell == TRAP) count++;
            }
        }
        return count;
    }

    int countHealth(const std::vector<std::vector<int>>& level) {
        int count = 0;
        for (const auto& row : level) {
            for (int cell : row) {
                if (cell == HEALTH) count++;
            }
        }
        return count;
    }

    int countKeys(const std::vector<std::vector<int>>& level) {
        int count = 0;
        for (const auto& row : level) {
            for (int cell : row) {
                if (cell == KEY) count++;
            }
        }
        return count;
    }
    void LevelGenerator::placeObjectsRL(std::vector<std::vector<int>>& level, int levelNum) {
        int size = level.size();
        sf::Vector2i playerPos, exitPos;
        std::vector<sf::Vector2i> emptyCells;
        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++) {
                if (level[y][x] == EMPTY) {
                    bool nearWall = false;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (level[y + dy][x + dx] == WALL) {
                                nearWall = true;
                                break;
                            }
                        }
                        if (nearWall) break;
                    }
                    if (!nearWall) {
                        emptyCells.emplace_back(x, y);
                    }
                }
            }
        }

        std::shuffle(emptyCells.begin(), emptyCells.end(), rng);

        if (!emptyCells.empty()) {
            auto playerPos = emptyCells.back();
            level[playerPos.y][playerPos.x] = PLAYER;
            emptyCells.pop_back();
        }

        if (!emptyCells.empty()) {
            sf::Vector2i exitPos;
            float maxDist = 0;
            for (const auto& pos : emptyCells) {
                float dist = sqrt(pow(pos.x - playerPos.x, 2) + pow(pos.y - playerPos.y, 2));
                if (dist > maxDist) {
                    maxDist = dist;
                    exitPos = pos;
                }
            }
            level[exitPos.y][exitPos.x] = EXIT;
            emptyCells.erase(std::remove(emptyCells.begin(), emptyCells.end(), exitPos), emptyCells.end());
        }
        int keysToPlace = std::min(1 + levelNum / 3, 2);
        for (int i = 0; i < keysToPlace && !emptyCells.empty(); i++) {
            auto keyPos = emptyCells.back();
            level[keyPos.y][keyPos.x] = KEY;
            emptyCells.pop_back();

            if (!emptyCells.empty()) {
                for (auto it = emptyCells.begin(); it != emptyCells.end(); ++it) {
                    if ((level[it->y - 1][it->x] == WALL && level[it->y + 1][it->x] == WALL) ||
                        (level[it->y][it->x - 1] == WALL && level[it->y][it->x + 1] == WALL)) {
                        level[it->y][it->x] = DOOR;
                        emptyCells.erase(it);
                        break;
                    }
                }
            }
        }

        int enemiesToPlace = std::min(3 + levelNum, static_cast<int>(emptyCells.size() * 0.2));
        for (int i = 0; i < enemiesToPlace && !emptyCells.empty(); i++) {
            auto pos = emptyCells.back();
            bool isStrong = (levelNum > 3) && (rand() % 100 < 30 + levelNum * 5);
            level[pos.y][pos.x] = isStrong ? STRONG_ENEMY : ENEMY;
            emptyCells.pop_back();
        }
        int trapsToPlace = std::min(2 + levelNum / 2, static_cast<int>(emptyCells.size() * 0.1));
        for (int i = 0; i < trapsToPlace && !emptyCells.empty(); i++) {
            auto pos = emptyCells.back();
            level[pos.y][pos.x] = TRAP;
            emptyCells.pop_back();
        }

        int healthToPlace = std::min(1 + levelNum / 4, static_cast<int>(emptyCells.size() * 0.05));
        for (int i = 0; i < healthToPlace && !emptyCells.empty(); i++) {
            auto pos = emptyCells.back();
            level[pos.y][pos.x] = HEALTH;
            emptyCells.pop_back();
        }

        ensureAccessibility(level);
    }

    void LevelGenerator::ensureAccessibility(std::vector<std::vector<int>>& level) {
        sf::Vector2i playerPos, exitPos;
        bool foundPlayer = false, foundExit = false;

        for (size_t y = 0; y < level.size(); y++) {
            for (size_t x = 0; x < level[y].size(); x++) {
                if (level[y][x] == PLAYER) {
                    playerPos = sf::Vector2i(x, y);
                    foundPlayer = true;
                }
                else if (level[y][x] == EXIT) {
                    exitPos = sf::Vector2i(x, y);
                    foundExit = true;
                }
            }
        }

        if (!foundPlayer || !foundExit) return;

        std::vector<sf::Vector2i> path = findPath(playerPos, exitPos, std::vector<Door>());
        if (path.empty()) {
            for (int i = 0; i < 10; i++) { 
                sf::Vector2i wallToRemove(
                    1 + rand() % (level[0].size() - 2),
                    1 + rand() % (level.size() - 2)
                );

                if (level[wallToRemove.y][wallToRemove.x] == WALL) {
                    level[wallToRemove.y][wallToRemove.x] = EMPTY;
                    path = findPath(playerPos, exitPos, std::vector<Door>());
                    if (!path.empty()) break;
                }
            }
        }
    }

    void createGuaranteedPath(std::vector<std::vector<int>>& level,
        const sf::Vector2i& start,
        const sf::Vector2i& end) {
        sf::Vector2i current = start;

        bool horizontalFirst = (rand() % 2) == 0;

        if (horizontalFirst) {
            int stepX = (end.x > current.x) ? 1 : -1;
            while (current.x != end.x) {
                current.x += stepX;
                if (level[current.y][current.x] == WALL) {
                    level[current.y][current.x] = EMPTY;
                }
            }

            int stepY = (end.y > current.y) ? 1 : -1;
            while (current.y != end.y) {
                current.y += stepY;
                if (level[current.y][current.x] == WALL) {
                    level[current.y][current.x] = EMPTY;
                }
            }
        }
        else {
            // Вертикаль
            int stepY = (end.y > current.y) ? 1 : -1;
            while (current.y != end.y) {
                current.y += stepY;
                if (level[current.y][current.x] == WALL) {
                    level[current.y][current.x] = EMPTY;
                }
            }

            // Горизонталь
            int stepX = (end.x > current.x) ? 1 : -1;
            while (current.x != end.x) {
                current.x += stepX;
                if (level[current.y][current.x] == WALL) {
                    level[current.y][current.x] = EMPTY;
                }
            }
        }
    }

    bool isOnPath(const std::vector<std::vector<int>>& level,
        const sf::Vector2i& point,
        const sf::Vector2i& start,
        const sf::Vector2i& end) {
        return (point.x >= std::min(start.x, end.x) && point.x <= std::max(start.x, end.x) &&
            point.y >= std::min(start.y, end.y) && point.y <= std::max(start.y, end.y));
    }

    float calculateLevelScore() const {
        float score = totalReward;
        int enemyKills = std::count_if(gameEvents.begin(), gameEvents.end(),
            [](const RLAgent::GameEvent& e) { return e.type == "enemy_killed"; });
        int trapsTriggered = std::count_if(gameEvents.begin(), gameEvents.end(),
            [](const RLAgent::GameEvent& e) { return e.type == "trap_triggered"; });
        int healthPicked = std::count_if(gameEvents.begin(), gameEvents.end(),
            [](const RLAgent::GameEvent& e) { return e.type == "health_picked"; });
        float balanceScore = (enemyKills - trapsTriggered + healthPicked) /
            static_cast<float>(gameEvents.size() + 1);

        return score + balanceScore * 0.5f;
    }
    void LevelGenerator::generateRooms(std::vector<std::vector<int>>& level) {
        int size = level.size();
        int roomCount = 3 + rand() % 3; 

        std::vector<sf::FloatRect> rooms;
        for (int i = 0; i < roomCount; i++) {
            int roomWidth = 3 + rand() % 4;
            int roomHeight = 3 + rand() % 4;
            int x = 1 + rand() % (size - roomWidth - 2);
            int y = 1 + rand() % (size - roomHeight - 2);
            sf::FloatRect newRoom(
                sf::Vector2f(static_cast<float>(x), static_cast<float>(y)), 
                sf::Vector2f(static_cast<float>(roomWidth), static_cast<float>(roomHeight))  
            );

            bool intersects = false;
            for (const auto& room : rooms) {
                if (newRoom.findIntersection(room).has_value()) {
                    intersects = true;
                    break;
                }
            }

            if (!intersects) {
                rooms.push_back(newRoom);
                for (int ry = y; ry < y + roomHeight; ry++) {
                    for (int rx = x; rx < x + roomWidth; rx++) {
                        if (ry == y || ry == y + roomHeight - 1 ||
                            rx == x || rx == x + roomWidth - 1) {
                            level[ry][rx] = WALL;
                        }
                        else {
                            level[ry][rx] = EMPTY;
                        }
                    }
                }
            }
        }
        for (size_t i = 1; i < rooms.size(); i++) {
            sf::Vector2i prevCenter(
                static_cast<int>(rooms[i - 1].position.x + rooms[i - 1].size.x / 2),
                static_cast<int>(rooms[i - 1].position.y + rooms[i - 1].size.y / 2)
            );
            sf::Vector2i currentCenter(
                static_cast<int>(rooms[i].position.x + rooms[i].size.x / 2),
                static_cast<int>(rooms[i].position.y + rooms[i].size.y / 2)
            );
            int stepX = (currentCenter.x > prevCenter.x) ? 1 : -1;
            for (int x = prevCenter.x; x != currentCenter.x; x += stepX) {
                if (level[prevCenter.y][x] == WALL) {
                    level[prevCenter.y][x] = EMPTY;
                }
            }

            int stepY = (currentCenter.y > prevCenter.y) ? 1 : -1;
            for (int y = prevCenter.y; y != currentCenter.y; y += stepY) {
                if (level[y][currentCenter.x] == WALL) {
                    level[y][currentCenter.x] = EMPTY;
                }
            }
        }
    }

    void placePlayerAndExit(std::vector<std::vector<int>>& level) {
        int size = level.size();
        std::vector<sf::Vector2i> emptyCells;

        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++) {
                if (level[y][x] == EMPTY) {
                    emptyCells.emplace_back(x, y);
                }
            }
        }

        if (!emptyCells.empty()) {
            int playerIndex = rand() % emptyCells.size();
            auto playerPos = emptyCells[playerIndex];
            level[playerPos.y][playerPos.x] = PLAYER;
            emptyCells.erase(emptyCells.begin() + playerIndex);

            if (!emptyCells.empty()) {
                sf::Vector2i exitPos;
                float maxDist = 0;
                for (const auto& pos : emptyCells) {
                    float dist = sqrt(pow(pos.x - playerPos.x, 2) + pow(pos.y - playerPos.y, 2));
                    if (dist > maxDist) {
                        maxDist = dist;
                        exitPos = pos;
                    }
                }
                level[exitPos.y][exitPos.x] = EXIT;
            }
        }
    }

    void LevelGenerator::addRandomBranches(std::vector<std::vector<int>>& level,
        const sf::Vector2i& start,
        const sf::Vector2i& end) {
        int size = level.size();
        int branchCount = 3 + rand() % 5;

        for (int i = 0; i < branchCount; ++i) {
            int x, y;
            if (rand() % 2) {
                x = start.x + rand() % abs(end.x - start.x);
                y = start.y;
            }
            else {
                x = end.x;
                y = start.y + rand() % abs(end.y - start.y);
            }

            if (level[y][x] != EMPTY) continue;
            int length = 2 + rand() % 4;
            int direction = rand() % 4;

            for (int j = 0; j < length; ++j) {
                switch (direction) {
                case 0: y--; break; 
                case 1: x++; break; 
                case 2: y++; break; 
                case 3: x--; break; 
                }

                if (x <= 0 || x >= size - 1 || y <= 0 || y >= size - 1) break;
                if (level[y][x] == WALL) level[y][x] = EMPTY;
            }
        }
    }



public:
    LevelGenerator() : rng(std::random_device()()) {


        objectWeights = {
             {"WALL", 0.1f},
             {"ENEMY", 0.3f},
             {"STRONG_ENEMY", 0.5f},
             {"TRAP", 0.4f},
             {"HEALTH", -0.2f},
             {"KEY", 0.2f},
             {"DOOR", 0.1f}
        };
        generatedLevels.push_back(firstLevel);

        for (int levelNum = 2; levelNum <= 6; levelNum++) {
            int baseSize = 10 + levelNum * 4; 
            int size = baseSize;
            std::vector<std::vector<int>> level(size, std::vector<int>(size, EMPTY));

            generateWalls(level, levelNum);
            placeObjectsRL(level, levelNum);
            generatedLevels.push_back(level);
        }
        generatedLevels.push_back(finalLevel);
        loadState();
    }

    void LevelGenerator::generateNewLevel(int levelNum) {
        std::vector<std::vector<int>> level;
        int attempts = 0;
        const int maxAttempts = 10;
        int baseSize = 10 + std::min(levelNum, 6) * 4; 
        do {
            int size = baseSize + 2;
            level = std::vector<std::vector<int>>(size, std::vector<int>(size, EMPTY));


            for (int i = 0; i < size; i++) {
                level[0][i] = WALL;
                level[size - 1][i] = WALL;
                level[i][0] = WALL;
                level[i][size - 1] = WALL;
            }


            generateRooms(level);
            placePlayerAndExit(level);
            addRandomBranches(level,
                sf::Vector2i(1, 1),
                sf::Vector2i(size - 2, size - 2));


            placeObjectsRL(level, levelNum);
            attempts++;

            if (attempts >= maxAttempts) {
                createSimpleLevel(level);
                break;
            }
        } while (!isLevelPassable(level));

        if (levelNum < generatedLevels.size()) {
            generatedLevels[levelNum] = level;
        }
        else {
            generatedLevels.push_back(level);
        }
    }
    void createSimpleLevel(std::vector<std::vector<int>>& level) {
        int size = level.size();

        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++) {
                if (y == 1 || y == size - 2 || x == 1 || x == size - 2) {
                    level[y][x] = WALL;
                }
                else {
                    level[y][x] = EMPTY;
                }
            }
        }


        level[size / 2][size / 2] = PLAYER;


        level[1][1] = EXIT;

        int enemies = 2 + rand() % 3;
        while (enemies-- > 0) {
            int x = 2 + rand() % (size - 4);
            int y = 2 + rand() % (size - 4);
            if (level[y][x] == EMPTY) {
                level[y][x] = ENEMY;
            }
        }
    }
    void adjustGenerationParameters(float levelScore) {

        if (levelScore > 0.7f) {
            currentDifficulty = std::min(2.0f, currentDifficulty * 1.1f);
        }
        else if (levelScore < 0.3f) {
            currentDifficulty = std::max(0.5f, currentDifficulty * 0.9f);
        }
        for (auto& event : gameEvents) {
            if (event.reward > 0) {
                objectWeights[event.type] = std::min(1.0f, objectWeights[event.type] * 1.05f);
            }
            else {
                objectWeights[event.type] = std::max(0.1f, objectWeights[event.type] * 0.95f);
            }
        }
    }

    void saveState() {
        rlAgent.saveToFile("rl_agent_state.txt");
    }

    void loadState() {
        rlAgent.loadFromFile("rl_agent_state.txt");
    }

    void recordEvent(const RLAgent::GameEvent& event) {
        gameEvents.push_back(event);
        totalReward += event.reward;
        rlAgent.update(event);
    }

    void endLevelEvaluation(int levelNum, float completionTime,
        int playerDeaths, int enemiesKilled,
        int trapsTriggered, int healthPicked) {
        float timeReward = 1.0f / (1.0f + completionTime / 120.0f);
        float deathPenalty = -0.5f * playerDeaths;
        float killReward = 0.1f * enemiesKilled;
        float trapPenalty = -0.3f * trapsTriggered;
        float healthReward = 0.2f * healthPicked;

        float levelReward = timeReward + deathPenalty + killReward + trapPenalty + healthReward;

        RLAgent::GameEvent event;
        event.type = "level_completed";
        event.reward = levelReward;
        event.additionalInfo = "level_" + std::to_string(levelNum);
        rlAgent.update(event);

        rlAgent.endEpisode();

        for (int levelNum = 0; levelNum <= 6; levelNum++) {
            generateNewLevel(levelNum);
        }
    }

    void updateDifficulty(float levelTime, int playerDeaths, int enemiesKilled,
        int trapsTriggered, int healthPicked);

    const std::vector<std::vector<int>>& getLevel(int index) const {
        if (index < 0 || index >= static_cast<int>(generatedLevels.size())) {
            return generatedLevels.front();
        }
        return generatedLevels[index];
    }

    int getLevelCount() const {
        return static_cast<int>(generatedLevels.size());
    }
};

void LevelGenerator::updateDifficulty(float levelTime, int playerDeaths, int enemiesKilled,
    int trapsTriggered, int healthPicked) {
    float timeReward = 1.0f / (1.0f + levelTime / 120.0f); 
    float deathPenalty = -0.5f * playerDeaths;
    float killReward = 0.1f * enemiesKilled;
    float trapPenalty = -0.3f * trapsTriggered;
    float healthReward = 0.2f * healthPicked;

    reward = timeReward + deathPenalty + killReward + trapPenalty + healthReward;

    playerSkill = 0.9f * playerSkill + 0.1f * (0.5f + 0.5f * reward);

    if (reward > 0.2f) { 
        currentDifficulty = std::min(2.0f, currentDifficulty * 1.1f);
    }
    else if (reward < -0.2f) {  
        currentDifficulty = std::max(0.5f, currentDifficulty * 0.9f);
    }


    for (auto& [obj, weight] : objectWeights) {
        if ((obj == "ENEMY" && killReward > 0) ||
            (obj == "TRAP" && trapPenalty < 0) ||
            (obj == "HEALTH" && healthReward > 0)) {
            weight = std::min(1.0f, weight * 1.05f);
        }
    }

    std::cout << "RL Update - Reward: " << reward
        << " | Difficulty: " << currentDifficulty
        << " | Player Skill: " << playerSkill << std::endl;
};


Wall createWall(b2World& world, const sf::Vector2f& position, const sf::Vector2f& size) {
    Wall wall;
    wall.shape.setSize(size);
    wall.shape.setFillColor(sf::Color::White);
    wall.shape.setOrigin(sf::Vector2f(size.x / 2, size.y / 2));
    wall.shape.setPosition(position);

    b2BodyDef wallDef;
    wallDef.type = b2_staticBody;
    wallDef.position.Set(position.x, position.y);
    wall.body = world.CreateBody(&wallDef);

    b2PolygonShape wallBox;
    wallBox.SetAsBox(size.x / 2, size.y / 2);

    b2FixtureDef fixtureDef;
    fixtureDef.shape = &wallBox;
    fixtureDef.density = 0.0f;
    fixtureDef.filter.categoryBits = WALL_CATEGORY;
    fixtureDef.filter.maskBits = PLAYER_CATEGORY | ENEMY_CATEGORY | BULLET_CATEGORY;
    wall.body->CreateFixture(&fixtureDef);

    return wall;
}

Pit createPit(const sf::Vector2f& position, const sf::Vector2f& size) {
    Pit pit;
    pit.shape.setSize(size);
    pit.shape.setFillColor(sf::Color::Blue);
    pit.shape.setOrigin(sf::Vector2f(size.x / 2, size.y / 20));
    pit.shape.setPosition(position);
    return pit;
}

Exit createExit(b2World& world, const sf::Vector2f& position, const sf::Vector2f& size) {
    Exit exit;
    exit.shape.setSize(size);
    exit.shape.setFillColor(sf::Color::Green);
    exit.shape.setOrigin(sf::Vector2f(size.x / 2, size.y / 2));
    exit.shape.setPosition(position);

    b2BodyDef exitDef;
    exitDef.type = b2_staticBody;
    exitDef.position.Set(position.x, position.y);
    exit.body = world.CreateBody(&exitDef);

    b2PolygonShape exitBox;
    exitBox.SetAsBox(size.x / 2, size.y / 2);

    b2FixtureDef fixtureDef;
    fixtureDef.shape = &exitBox;
    fixtureDef.isSensor = true; 
    fixtureDef.filter.categoryBits = EXIT_CATEGORY;
    fixtureDef.filter.maskBits = PLAYER_CATEGORY;
    exit.body->CreateFixture(&fixtureDef);

    return exit;
}
HealthPickup createHealthPickup(b2World& world, const sf::Vector2f& position) {
    HealthPickup health;
    health.shape = sf::CircleShape(10.0f, 30);
    health.shape.setFillColor(sf::Color::Red);
    health.shape.setOrigin(sf::Vector2f(10.0f, 10.0f));
    health.shape.setPosition(position);
    health.pulseSpeed = 1.0f + (rand() % 100) * 0.01f;
    health.pulseSize = 0.2f;
    health.pulseTime = static_cast<float>(rand() % 100) * 0.01f * 2 * b2_pi;

    b2BodyDef healthDef;
    healthDef.type = b2_staticBody;
    healthDef.position.Set(position.x, position.y);
    health.body = world.CreateBody(&healthDef);

    b2CircleShape circle;
    circle.m_radius = 10.0f;

    b2FixtureDef fixtureDef;
    fixtureDef.shape = &circle;
    fixtureDef.isSensor = true;
    fixtureDef.filter.categoryBits = HEALTH_CATEGORY;
    fixtureDef.filter.maskBits = PLAYER_CATEGORY; 
    health.body->CreateFixture(&fixtureDef);

    return health;
}

Trap createTrap(b2World& world, const sf::Vector2f& position) {
    Trap trap;
    trap.shape.setPointCount(3);
    trap.shape.setPoint(0, sf::Vector2f(0, -10));
    trap.shape.setPoint(1, sf::Vector2f(8, 10));
    trap.shape.setPoint(2, sf::Vector2f(-8, 10));
    trap.shape.setFillColor(sf::Color::Red);
    trap.shape.setOrigin(sf::Vector2f(0, 0)); 
    trap.shape.setPosition(position);

    // Физическое тело
    b2BodyDef trapDef;
    trapDef.type = b2_staticBody;
    trapDef.position.Set(position.x, position.y);
    trap.body = world.CreateBody(&trapDef);

    b2PolygonShape trapShape;
    b2Vec2 vertices[3];
    vertices[0].Set(0, -10);
    vertices[1].Set(8, 10);
    vertices[2].Set(-8, 10);
    trapShape.Set(vertices, 3);

    b2FixtureDef fixtureDef;
    fixtureDef.shape = &trapShape;
    fixtureDef.isSensor = true;
    fixtureDef.filter.categoryBits = TRAP_CATEGORY;
    fixtureDef.filter.maskBits = PLAYER_CATEGORY;
    trap.body->CreateFixture(&fixtureDef);

    return trap;
}

Key createKey(b2World& world, const sf::Vector2f& position) {
    Key key;
    key.shape.setSize(sf::Vector2f(10.0f, 20.0f));
    key.shape.setFillColor(sf::Color::Yellow);
    key.shape.setOrigin(sf::Vector2f(5.0f, 10.0f));
    key.shape.setPosition(position);

    b2BodyDef keyDef;
    keyDef.type = b2_staticBody;
    keyDef.position.Set(position.x, position.y);
    key.body = world.CreateBody(&keyDef);

    b2PolygonShape keyShape;
    keyShape.SetAsBox(5.0f, 10.0f);

    b2FixtureDef fixtureDef;
    fixtureDef.shape = &keyShape;
    fixtureDef.isSensor = true;
    fixtureDef.filter.categoryBits = KEY_CATEGORY;
    fixtureDef.filter.maskBits = PLAYER_CATEGORY;
    key.body->CreateFixture(&fixtureDef);

    return key;
}

// Функция создания двери
Door createDoor(b2World& world, const sf::Vector2f& position, const sf::Vector2f& size) {
    Door door;
    door.shape.setSize(size);
    door.shape.setFillColor(sf::Color(139, 69, 19)); 
    door.shape.setOrigin(sf::Vector2f(size.x / 2, size.y / 2));
    door.shape.setPosition(position);

    b2BodyDef doorDef;
    doorDef.type = b2_staticBody;
    doorDef.position.Set(position.x, position.y);
    door.body = world.CreateBody(&doorDef);

    b2PolygonShape doorShape;
    doorShape.SetAsBox(size.x / 2, size.y / 2);
    b2FixtureDef fixtureDef;
    fixtureDef.shape = &doorShape;
    fixtureDef.density = 0.0f;
    fixtureDef.filter.categoryBits = WALL_CATEGORY;
    fixtureDef.filter.maskBits = PLAYER_CATEGORY | ENEMY_CATEGORY | BULLET_CATEGORY;
    door.body->CreateFixture(&fixtureDef);

    return door;
}

bool isWalkable(int x, int y, const std::vector<Door>& doors) {
    if (x < 0 || y < 0 || x >= currentLevelMap[0].size() || y >= currentLevelMap.size())
        return false;
    for (const auto& door : doors) {
        sf::Vector2f doorPos = door.shape.getPosition();
        int doorX = static_cast<int>(doorPos.x / cellSize);
        int doorY = static_cast<int>(doorPos.y / cellSize);

        if (doorX == x && doorY == y && !door.opened) {
            return false;
        }
    }

    return currentLevelMap[y][x] != WALL && currentLevelMap[y][x] != PIT;
}

std::vector<sf::Vector2i> findPath(const sf::Vector2i& start, const sf::Vector2i& end, const std::vector<Door>& doors) {
    std::vector<sf::Vector2i> path;

    if (!isWalkable(end.x, end.y, doors)) {
        return path;
    }

    auto heuristic = [](const sf::Vector2i& a, const sf::Vector2i& b) {
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
        };

    std::vector<Node*> openList;
    std::vector<Node*> closedList;
    std::unordered_map<int, std::unordered_map<int, bool>> openMap;
    std::unordered_map<int, std::unordered_map<int, bool>> closedMap;

    Node* startNode = new Node(start.x, start.y);
    openList.push_back(startNode);
    openMap[start.x][start.y] = true;

    while (!openList.empty()) {
        auto it = std::min_element(openList.begin(), openList.end(),
            [](const Node* a, const Node* b) { return a->getF() < b->getF(); });
        Node* current = *it;
        openList.erase(it);
        openMap[current->x][current->y] = false;
        closedList.push_back(current);
        closedMap[current->x][current->y] = true;

        if (current->x == end.x && current->y == end.y) {
            while (current != nullptr) {
                path.emplace_back(current->x, current->y);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
            break;
        }


        const std::vector<sf::Vector2i> directions = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
        for (const auto& dir : directions) {
            int newX = current->x + dir.x;
            int newY = current->y + dir.y;
            if (!isWalkable(newX, newY, doors) || closedMap[newX][newY]) {
                continue;
            }

            float newG = current->g + 1;
            Node* successor = nullptr;
            auto found = std::find_if(openList.begin(), openList.end(),
                [newX, newY](const Node* n) { return n->x == newX && n->y == newY; });

            if (found != openList.end()) {
                successor = *found;
                if (newG >= successor->g) {
                    continue;
                }
            }
            else {
                successor = new Node(newX, newY, current);
                openList.push_back(successor);
                openMap[newX][newY] = true;
            }

            successor->g = newG;
            successor->h = heuristic(sf::Vector2i(newX, newY), end);
            successor->parent = current;
        }
    }
    for (auto node : openList) delete node;
    for (auto node : closedList) delete node;

    return path;
}

std::vector<sf::Vector2i> createPatrolPath(int startX, int startY, const std::vector<Door>& doors) {
    std::vector<sf::Vector2i> path;
    const int maxPatrolPoints = 3 + rand() % 3;
    const int maxAttempts = 20; 

    int currentX = startX;
    int currentY = startY;
    path.emplace_back(currentX, currentY);

    for (int i = 1; i < maxPatrolPoints; ++i) {
        int attempts = 0;
        while (attempts < maxAttempts) {
            int dir = rand() % 6; 

            if (i > 1 && dir >= 4) {
                sf::Vector2i prevDir = path[i - 1] - path[i - 2];
                if (prevDir.x != 0 || prevDir.y != 0) {
                    path.emplace_back(path.back().x + prevDir.x, path.back().y + prevDir.y);
                    break;
                }
            }

            int newX = currentX;
            int newY = currentY;

            switch (dir % 4) {
            case 0: newY--; break;
            case 1: newX++; break;
            case 2: newY++; break; 
            case 3: newX--; break; 
            }

            if (i > 1 && newX == path[i - 2].x && newY == path[i - 2].y) {
                attempts++;
                continue;
            }

            if (isWalkable(newX, newY, doors)) {
                path.emplace_back(newX, newY);
                currentX = newX;
                currentY = newY;
                break;
            }
            attempts++;
        }
        if (attempts >= maxAttempts) {
            break;
        }
    }

    return path;
}


class ContactListener : public b2ContactListener {
    std::vector<Bullet*>& bullets;
    std::vector<Enemy*>& enemies;
    std::vector<Trap>& traps;
    Player& player;
    Exit& exit;
    std::vector<HealthPickup>& healthPickups;
    bool& levelCompleted;
    b2World& world;
    std::vector<Key>& keys;
    std::vector<Door>& doors;
    int& playerDeaths;
    int& enemiesKilled;
    int& trapsTriggered;
    int& healthPicked;

public:
    ContactListener(std::vector<Bullet*>& bullets, std::vector<Enemy*>& enemies,
        Player& player, Exit& exit, std::vector<HealthPickup>& healthPickups, bool& levelCompleted, b2World& world, std::vector<Trap>& traps, std::vector<Key>& keys, std::vector<Door>& doors, int& pd, int& ek, int& tt, int& hp)
        : bullets(bullets), enemies(enemies), player(player),
        exit(exit), healthPickups(healthPickups), levelCompleted(levelCompleted), world(world), traps(traps), keys(keys), doors(doors), playerDeaths(pd), enemiesKilled(ek), trapsTriggered(tt), healthPicked(hp) {}

    ContactListener& operator=(const ContactListener&) = delete;
    LevelGenerator levelGenerator;
    void BeginContact(b2Contact* contact) override {
        b2Fixture* fixtureA = contact->GetFixtureA();
        b2Fixture* fixtureB = contact->GetFixtureB();

        auto createEvent = [this](const std::string& type, float reward, sf::Vector2f pos) {
            RLAgent::GameEvent event;
            event.type = type;
            event.reward = reward;
            event.position = pos;
            levelGenerator.recordEvent(event);
            };
        for (auto* bullet : bullets) {
            b2Fixture* bulletFixture = bullet->body->GetFixtureList();

            if (fixtureA == bulletFixture || fixtureB == bulletFixture) {
                bullet->toDestroy = true;
                for (auto* enemy : enemies) {
                    if (!enemy->body || enemy->health <= 0) continue;

                    b2Fixture* enemyFixture = enemy->body->GetFixtureList();
                    if (fixtureA == enemyFixture || fixtureB == enemyFixture) {
                        enemy->health--; 
                        enemy->stunTimer = 0.0f; 
                        if (enemy->health <= 0) {
                            enemy->toDestroy = true;
                            RLAgent::GameEvent event;
                            createEvent(enemy->isStrong ? "strong_enemy_killed" : "enemy_killed",
                                enemy->isStrong ? 1.0f : 0.5f,
                                enemy->shape.getPosition());
                            event.position = enemy->shape.getPosition();
                            levelGenerator.recordEvent(event);
                            enemiesKilled++;
                        }
                        break;
                    }
                }
            }
        }
        for (auto* enemy : enemies) {
            if (!enemy->body || enemy->stunTimer > 0) continue; 

            b2Fixture* enemyFixture = enemy->body->GetFixtureList();
            if ((fixtureA == enemyFixture && fixtureB->GetBody() == player.body) ||
                (fixtureB == enemyFixture && fixtureA->GetBody() == player.body)) {


                int damage = enemy->isStrong ? 2 : 1; 

                while (damage > 0 && player.bonusLives > 0) {
                    player.bonusLives--;
                    damage--;
                }

                while (damage > 0 && player.lives > 0) {
                    player.lives--;
                    damage--;
                }

                enemy->stunTimer = 1.0f;

                if (player.lives <= 0 && player.bonusLives <= 0) {
                    player.lives = 0;
                    levelCompleted = true; 
                }
                break;
            }
        }
        if ((fixtureA->GetBody() == player.body && fixtureB->GetBody() == exit.body) ||
            (fixtureB->GetBody() == player.body && fixtureA->GetBody() == exit.body)) {
            levelCompleted = true;
        }

        for (auto& health : healthPickups) {
            if (health.active &&
                ((fixtureA->GetBody() == player.body && fixtureB->GetBody() == health.body) ||
                    (fixtureB->GetBody() == player.body && fixtureA->GetBody() == health.body))) {
                health.active = false;
                if (player.lives < 3) {
                    player.lives++;
                    createEvent("health_picked", 0.3f, health.shape.getPosition());
                    healthPicked++;
                }
                else {
                    player.bonusLives++;
                    createEvent("health_picked", 0.3f, health.shape.getPosition());
                    healthPicked++;
                }
                break;
            }
        }
        for (auto& trap : traps) {
            if (trap.active &&
                ((fixtureA->GetBody() == player.body && fixtureB->GetBody() == trap.body) ||
                    (fixtureB->GetBody() == player.body && fixtureA->GetBody() == trap.body))) {
                trap.active = false;

                if (player.bonusLives > 0) {
                    player.bonusLives--;
                    createEvent("trap_triggered", -0.7f, trap.shape.getPosition());
                    trapsTriggered++;
                }
                else if (player.lives > 0) {
                    player.lives--;
                    createEvent("trap_triggered", -0.7f, trap.shape.getPosition());
                    trapsTriggered++;
                }

                if (player.lives <= 0 && player.bonusLives <= 0) {
                    player.lives = 0;
                    levelCompleted = true;
                    createEvent("trap_triggered", -0.7f, trap.shape.getPosition());
                    trapsTriggered++;
                }
                break;
            }
        }
        for (auto& key : keys) {
            if (!key.collected &&
                ((fixtureA->GetBody() == player.body && fixtureB->GetBody() == key.body) ||
                    (fixtureB->GetBody() == player.body && fixtureA->GetBody() == key.body))) {
                key.collected = true;
                player.keys++;
                break;
            }
        }

        for (auto& door : doors) {
            if (!door.opened && player.keys > 0 &&
                ((fixtureA->GetBody() == player.body && fixtureB->GetBody() == door.body) ||
                    (fixtureB->GetBody() == player.body && fixtureA->GetBody() == door.body))) {
                door.opened = true;
                player.keys--;
                door.toDestroy = true;  
                door.shape.setFillColor(sf::Color(139, 69, 19, 128));
                break;
            }
        }
    }
};

void updateTraps(std::vector<Trap>& traps, b2World& world, float deltaTime) {
    for (auto it = traps.begin(); it != traps.end(); ) {
        if (!it->active) {
            world.DestroyBody(it->body);
            it = traps.erase(it);
        }
        else {
            it->shape.rotate(sf::degrees(it->rotationSpeed * deltaTime));
            ++it;
        }
    }
}

void updatePlayer(Player& player, float deltaTime) { 
    b2Vec2 velocity(0.0f, 0.0f);

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) {
        velocity.y -= player.speed;
        player.angle = 0.0f;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) {
        velocity.y += player.speed;
        player.angle = 180.0f;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) {
        velocity.x -= player.speed;
        player.angle = 270.0f;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) {
        velocity.x += player.speed;
        player.angle = 90.0f;
    }

    player.body->SetLinearVelocity(velocity);
    player.shape.setPosition(sf::Vector2f(player.body->GetPosition().x, player.body->GetPosition().y));
    player.shape.setRotation(sf::degrees(player.angle));
    player.directionArc.setPosition(sf::Vector2f(player.shape.getPosition()));
    player.directionArc.setRotation(sf::degrees(player.angle));
    player.lastShotTime += deltaTime;
}

void updateHearts(std::vector<Heart>& hearts, const Player& player, int bonusHearts) {
    hearts.clear();
    for (int i = 0; i < player.lives; ++i) {
        Heart h;
        h.shape = sf::CircleShape(8.0f, 30);
        h.shape.setFillColor(sf::Color::Red);
        h.shape.setOrigin(sf::Vector2f(8.0f, 8.0f));
        h.isBonus = false;
        hearts.push_back(h);
    }
    for (int i = 0; i < bonusHearts; ++i) {
        Heart h;
        h.shape = sf::CircleShape(8.0f, 30);
        h.shape.setFillColor(sf::Color::Yellow);
        h.shape.setOrigin(sf::Vector2f(8.0f, 8.0f));
        h.isBonus = true;
        hearts.push_back(h);
    }
    for (size_t i = 0; i < hearts.size(); ++i) {
        hearts[i].shape.setPosition(sf::Vector2f(20.0f + i * 21.0f, 20.0f));
    }
}

void updateHealthPickups(std::vector<HealthPickup>& healthPickups, b2World& world) {
    for (auto it = healthPickups.begin(); it != healthPickups.end(); ) {
        if (!it->active) {
            world.DestroyBody(it->body);
            it = healthPickups.erase(it);
        }
        else {
            ++it;
        }
    }
}

void updateHealthPickupsAnimation(std::vector<HealthPickup>& healthPickups, float deltaTime) {
    for (auto& health : healthPickups) {
        if (health.active) {
            health.pulseTime += deltaTime * health.pulseSpeed;
            float scale = 1.0f + sin(health.pulseTime) * health.pulseSize;
            health.shape.setScale(sf::Vector2f(scale, scale));
        }
    }
}

void updateEnemies(std::vector<Enemy*>& enemies, const sf::Vector2f& playerPosition, b2World& world, float deltaTime, const std::vector<Door>& doors) {
    sf::Vector2i playerCell(static_cast<int>(playerPosition.x / cellSize),
        static_cast<int>(playerPosition.y / cellSize));

    for (auto it = enemies.begin(); it != enemies.end(); ) {
        Enemy* enemy = *it;
        if (enemy->toDestroy) {
            if (enemy->body) {
                world.DestroyBody(enemy->body);
                enemy->body = nullptr;
            }
            delete enemy;
            it = enemies.erase(it);
            continue;
        }

        if (enemy->health <= 0) {
            enemy->toDestroy = true;
            ++it;
            continue;
        }

        if (!enemy->body) {
            ++it;
            continue;
        }
        if (enemy->stunTimer > 0) {
            enemy->stunTimer -= deltaTime;
            enemy->body->SetLinearVelocity(b2Vec2(0, 0));
            enemy->shape.setFillColor(sf::Color::Cyan);
            ++it;
            continue;
        }
        else {
            enemy->shape.setFillColor(enemy->isStrong ? sf::Color::Magenta : sf::Color::Green);
        }

        sf::Vector2f enemyPos = enemy->shape.getPosition();
        sf::Vector2i enemyCell(static_cast<int>(enemyPos.x / cellSize),
            static_cast<int>(enemyPos.y / cellSize));

        float distanceToPlayer = std::sqrt(std::pow(playerCell.x - enemyCell.x, 2) +
            std::pow(playerCell.y - enemyCell.y, 2));
        bool hasLineOfSight = false;
        if (distanceToPlayer <= 5.0f) {
            hasLineOfSight = true;
            sf::Vector2i dir = playerCell - enemyCell;
            int steps = std::max(std::abs(dir.x), std::abs(dir.y));

            if (steps > 0) {
                float dx = static_cast<float>(dir.x) / steps;
                float dy = static_cast<float>(dir.y) / steps;

                for (int i = 1; i < steps; ++i) {
                    int checkX = enemyCell.x + static_cast<int>(dx * i);
                    int checkY = enemyCell.y + static_cast<int>(dy * i);

                    if (!isWalkable(checkX, checkY, doors)) {
                        hasLineOfSight = false;
                        break;
                    }
                }
            }
        }
        if (hasLineOfSight) {
            enemy->hasSeenPlayer = true;
            enemy->isPursuing = true;
            enemy->lastSeenPlayerTime = 3.0f; 
        }
        else {
            enemy->lastSeenPlayerTime -= deltaTime;
            if (enemy->lastSeenPlayerTime <= 0) {
                enemy->isPursuing = false;
                enemy->path.clear();
            }
        }
        enemy->patrolChangeTimer += deltaTime;
        if (enemy->patrolChangeTimer >= 10.0f) {
            enemy->patrolPath = createPatrolPath(enemyCell.x, enemyCell.y, doors);
            enemy->currentPatrolPoint = 0;
            enemy->patrolChangeTimer = 0.0f;
        }

        if (enemy->isPursuing && enemy->hasSeenPlayer) {
            enemy->recalculatePathTimer += deltaTime;
            if (enemy->recalculatePathTimer >= 0.1f || enemy->path.empty()) {
                enemy->path = findPath(enemyCell, playerCell, doors);
                enemy->recalculatePathTimer = 0.0f;
            }

            if (!enemy->path.empty() && enemy->path.size() > 1) {
                sf::Vector2i nextCell = enemy->path[1];
                if (isWalkable(nextCell.x, nextCell.y, doors)) {
                    sf::Vector2f targetPos(nextCell.x * cellSize + cellSize / 2,
                        nextCell.y * cellSize + cellSize / 2);

                    b2Vec2 direction(targetPos.x - enemyPos.x, targetPos.y - enemyPos.y);
                    direction.Normalize();
                    enemy->body->SetLinearVelocity(enemy->speed * direction);
                }
                else {
                    enemy->isPursuing = false;
                    enemy->path.clear();
                }
            }
            else {
                b2Vec2 direction(playerPosition.x - enemyPos.x, playerPosition.y - enemyPos.y);
                direction.Normalize();
                enemy->body->SetLinearVelocity(enemy->speed * direction);
            }
        }
        else {
            enemy->path.clear();
            if (enemy->patrolPath.empty()) {
                enemy->patrolPath = createPatrolPath(enemyCell.x, enemyCell.y, doors);
                enemy->currentPatrolPoint = 0;
            }

            if (!enemy->patrolPath.empty()) {
                sf::Vector2i targetCell = enemy->patrolPath[enemy->currentPatrolPoint];
                sf::Vector2f targetPos(targetCell.x * cellSize + cellSize / 2,
                    targetCell.y * cellSize + cellSize / 2);

                float distanceToTarget = std::sqrt(std::pow(targetPos.x - enemyPos.x, 2) +
                    std::pow(targetPos.y - enemyPos.y, 2));

                if (distanceToTarget < 5.0f) {
                    enemy->idleTimer += deltaTime;
                    enemy->body->SetLinearVelocity(b2Vec2(0, 0));

                    if (enemy->idleTimer >= 1.0f + (rand() % 100) * 0.02f) {
                        enemy->idleTimer = 0.0f;
                        enemy->currentPatrolPoint = (enemy->currentPatrolPoint + 1) % enemy->patrolPath.size();
                    }
                }
                else {
                    b2Vec2 direction(targetPos.x - enemyPos.x, targetPos.y - enemyPos.y);
                    direction.Normalize();
                    enemy->body->SetLinearVelocity(enemy->speed * 0.7f * direction);
                }
            }
        }
        if (enemy->body) {
            enemy->shape.setPosition(sf::Vector2f(enemy->body->GetPosition().x, enemy->body->GetPosition().y));
        }

        ++it;
    }
}

void clearGameObjects(b2World& world,
    std::vector<Wall>& walls,
    std::vector<Pit>& pits,
    std::vector<Enemy*>& enemies,
    std::vector<Bullet*>& bullets,
    std::vector<HealthPickup>& healthPickups,
    std::vector<Trap>& traps,
    std::vector<Key>& keys,
    std::vector<Door>& doors) {
    for (auto* bullet : bullets) {
        world.DestroyBody(bullet->body);
        delete bullet;
    }
    bullets.clear();
    for (auto* enemy : enemies) {
        if (enemy->body) {  
            world.DestroyBody(enemy->body);
        }
        delete enemy;
    }
    enemies.clear();

    for (auto& wall : walls) {
        world.DestroyBody(wall.body);
    }
    walls.clear();

    for (auto& trap : traps) {
        world.DestroyBody(trap.body);
    }
    traps.clear();

    for (auto& key : keys) {
        world.DestroyBody(key.body);
    }
    keys.clear();

    for (auto& door : doors) {
        world.DestroyBody(door.body);
    }
    doors.clear();

    for (auto& health : healthPickups) {
        world.DestroyBody(health.body);
    }
    healthPickups.clear();

    pits.clear(); 
}

void resetContactListener(b2World& world, ContactListener*& listener,
    std::vector<Bullet*>& bullets, std::vector<Enemy*>& enemies,
    Player& player, Exit& exit, std::vector<HealthPickup>& healthPickups,
    bool& levelCompleted, std::vector<Trap>& traps,
    std::vector<Key>& keys, std::vector<Door>& doors,
    int& pd, int& ek, int& tt, int& hp) {

    world.SetContactListener(nullptr);
    delete listener;
    listener = new ContactListener(bullets, enemies, player, exit, healthPickups,
        levelCompleted, world, traps, keys, doors, pd, ek, tt, hp);
    world.SetContactListener(listener);
}

void createBullet(std::vector<Bullet*>& bullets, Player& player, b2World& world) {
    Bullet* bullet = new Bullet();
    bullet->shape = sf::CircleShape(5.0f);
    bullet->shape.setFillColor(sf::Color::Red);
    bullet->shape.setOrigin(sf::Vector2f(5.0f, 5.0f));

    float radianAngle = (player.angle - 90.0f) * b2_pi / 180.0f;
    float offset = 22.0f;
    sf::Vector2f startPos = player.shape.getPosition() + sf::Vector2f(cos(radianAngle) * offset, sin(radianAngle) * offset);
    bullet->shape.setPosition(startPos);
    bullet->direction = b2Vec2(cos(radianAngle), sin(radianAngle));

    b2BodyDef bulletDef;
    bulletDef.type = b2_dynamicBody;
    bulletDef.position.Set(startPos.x, startPos.y);
    bulletDef.bullet = true;
    bullet->body = world.CreateBody(&bulletDef);

    b2CircleShape circle;
    circle.m_radius = 5.0f;

    b2FixtureDef fixtureDef;
    fixtureDef.shape = &circle;
    fixtureDef.density = 1.0f;
    fixtureDef.isSensor = false;
    fixtureDef.filter.categoryBits = BULLET_CATEGORY;
    fixtureDef.filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY;  
    bullet->body->CreateFixture(&fixtureDef);

    bullet->body->SetLinearVelocity(60.0f * bullet->direction);
    bullets.push_back(bullet);
}

void updateBullets(std::vector<Bullet*>& bullets, b2World& world) {
    for (auto it = bullets.begin(); it != bullets.end();) {
        Bullet* bullet = *it;

        bullet->shape.setPosition(sf::Vector2f(bullet->body->GetPosition().x, bullet->body->GetPosition().y));

        b2Filter filter;
        filter.categoryBits = BULLET_CATEGORY;
        filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY;
        bullet->body->GetFixtureList()->SetFilterData(filter);
        b2Vec2 vel = bullet->body->GetLinearVelocity();
        bullet->distanceTravelled += sqrt(vel.x * vel.x + vel.y * vel.y) * (1.0f / 60.0f);


        if (bullet->toDestroy || bullet->distanceTravelled > bullet->maxDistance) {
            world.DestroyBody(bullet->body);
            delete bullet;
            it = bullets.erase(it);
        }
        else {
            ++it;
        }
    }
}

void updateDoors(std::vector<Door>& doors, b2World& world) {
    for (auto it = doors.begin(); it != doors.end(); ) {
        if (it->toDestroy) {
            world.DestroyBody(it->body);
            it = doors.erase(it);
        }
        else {
            ++it;
        }
    }
}

void drawKeys(sf::RenderWindow& window, const Player& player) {
    for (int i = 0; i < player.keys; ++i) {
        sf::RectangleShape keyShape(sf::Vector2f(15.0f, 30.0f));
        keyShape.setFillColor(sf::Color::Yellow);
        keyShape.setOutlineColor(sf::Color::Black);
        keyShape.setOutlineThickness(1.0f);
        keyShape.setPosition(sf::Vector2f(window.getSize().x - 40.0f - i * 25.0f, 20.0f));
        window.draw(keyShape);
    }
}

void parseMap(const std::vector<std::vector<int>>& map, float cellSize,
    b2World& world, Player& player, std::vector<Wall>& walls,
    std::vector<Pit>& pits, std::vector<Enemy*>& enemies, Exit& exit, std::vector<HealthPickup>& healthPickups, std::vector<Trap>& traps, std::vector<Key>& keys, std::vector<Door>& doors) {
    currentLevelMap = map;
    for (size_t y = 0; y < map.size(); ++y) {
        for (size_t x = 0; x < map[y].size(); ++x) {
            sf::Vector2f position(x * cellSize + cellSize / 2, y * cellSize + cellSize / 2);
            sf::Vector2f size(cellSize, cellSize);

            switch (map[y][x]) {
            case WALL: {
                walls.push_back(createWall(world, position, size));
                break;
            }

            case PIT: {
                pits.push_back(createPit(position, size));
                break;
            }

            case PLAYER: {
                player.shape.setPosition(position);
                player.body->SetTransform(b2Vec2(position.x, position.y), 0);

                b2Fixture* playerFixture = player.body->GetFixtureList();
                b2Filter filter = playerFixture->GetFilterData();
                filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY |
                    KEY_CATEGORY | HEALTH_CATEGORY | TRAP_CATEGORY | EXIT_CATEGORY;
                playerFixture->SetFilterData(filter);
                break;
            }

            case EXIT: {
                exit = createExit(world, position, size);
                break;
            }

            case HEALTH: {
                healthPickups.push_back(createHealthPickup(world, position));
                break;
            }

            case KEY: {
                keys.push_back(createKey(world, position));
                break;
            }
            case DOOR: {
                doors.push_back(createDoor(world, position, sf::Vector2f(cellSize, cellSize)));
                break;
            }

            case ENEMY: {
                Enemy* enemy = new Enemy();
                enemy->shape = sf::CircleShape(12.0f, 30);
                enemy->shape.setFillColor(sf::Color::Green);
                enemy->shape.setOrigin(sf::Vector2f(12.0f, 12.0f)); 
                enemy->speed = 150.0f;
                enemy->health = 3;
                enemy->isStrong = false;

                b2BodyDef enemyDef;
                enemyDef.type = b2_dynamicBody;
                enemyDef.position.Set(position.x, position.y);
                enemy->body = world.CreateBody(&enemyDef);

                b2CircleShape enemyShape;
                enemyShape.m_radius = 12.0f;

                b2FixtureDef enemyFixture;
                enemyFixture.shape = &enemyShape;
                enemyFixture.density = 1.0f;
                enemyFixture.friction = 0.3f; 
                enemyFixture.filter.categoryBits = ENEMY_CATEGORY;
                enemyFixture.filter.maskBits = PLAYER_CATEGORY | WALL_CATEGORY | BULLET_CATEGORY | ENEMY_CATEGORY;
                enemy->body->CreateFixture(&enemyFixture);

                enemies.push_back(enemy);
                break;
            }
            case TRAP: {
                traps.push_back(createTrap(world, position));
                std::cout << "Trap created at: " << position.x << ", " << position.y << std::endl;
                break;
            }

            case STRONG_ENEMY: {
                Enemy* enemy = new Enemy();
                enemy->shape = sf::CircleShape(15.0f, 30);
                enemy->shape.setFillColor(sf::Color::Magenta);
                enemy->shape.setOrigin(sf::Vector2f(15.0f, 15.0f)); 
                enemy->speed = 75.0f;
                enemy->health = 5;
                enemy->isStrong = true;

                b2BodyDef enemyDef;
                enemyDef.type = b2_dynamicBody;
                enemyDef.position.Set(position.x, position.y);
                enemy->body = world.CreateBody(&enemyDef);

                b2CircleShape enemyShape;
                enemyShape.m_radius = 15.0f;

                b2FixtureDef enemyFixture;
                enemyFixture.shape = &enemyShape;
                enemyFixture.density = 1.0f;
                enemyFixture.friction = 0.3f;
                enemyFixture.filter.categoryBits = ENEMY_CATEGORY;
                enemyFixture.filter.maskBits = PLAYER_CATEGORY | WALL_CATEGORY | BULLET_CATEGORY | ENEMY_CATEGORY;
                enemy->body->CreateFixture(&enemyFixture);

                enemies.push_back(enemy);
                break;
            }
            }
        }
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Roguelike");
    window.setFramerateLimit(60);
    LevelGenerator levelGenerator;
    levelGenerator.loadState();
    b2World world(b2Vec2(0, 0));
    auto levelStartTime = std::chrono::steady_clock::now();
    int playerDeaths = 0;
    int enemiesKilled = 0;
    int trapsTriggered = 0;
    int healthPicked = 0;
    sf::Music music;
    if (!music.openFromFile("assets/ambient.mp3")) {
        std::cerr << "Failed to load music" << std::endl;
    }
    music.setLooping(true);
    music.play();
    music.setVolume(10.0f);
    Player player;
    player.shape = sf::CircleShape(12.0f, 30);
    player.shape.setFillColor(sf::Color::Blue);
    player.shape.setOrigin(sf::Vector2f(12.0f, 12.0f));
    player.shape.setPosition(sf::Vector2f(400.0f, 300.0f));
    player.lives = 3;
    player.speed = 100.0f;
    player.angle = 0.0f;

    player.directionArc.setPointCount(3);
    player.directionArc.setPoint(0, sf::Vector2f(0.0f, -16.0f));
    player.directionArc.setPoint(1, sf::Vector2f(8.0f, 0.0f));
    player.directionArc.setPoint(2, sf::Vector2f(-8.0f, 0.0f));
    player.directionArc.setFillColor(sf::Color(255, 0, 0, 150));
    player.directionArc.setOrigin(sf::Vector2f(0.0f, 0.0f));

    b2BodyDef playerDef;
    playerDef.type = b2_dynamicBody;
    playerDef.position.Set(400.0f, 300.0f);
    player.body = world.CreateBody(&playerDef);

    b2CircleShape playerShape;
    playerShape.m_radius = 12.0f;
    b2FixtureDef playerFixture;
    playerFixture.shape = &playerShape;
    playerFixture.density = 1.0f;
    playerFixture.filter.categoryBits = PLAYER_CATEGORY;
    playerFixture.filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY |
        KEY_CATEGORY | HEALTH_CATEGORY | TRAP_CATEGORY | EXIT_CATEGORY;
    player.body->CreateFixture(&playerFixture);

    std::vector<Trap> traps;
    std::vector<Wall> walls;
    std::vector<Pit> pits;
    std::vector<Enemy*> enemies;
    std::vector<Bullet*> bullets;
    std::vector<Heart> hearts;
    std::vector<Key> keys;
    std::vector<Door> doors;
    const float cellSize = 32.0f;
    Exit exit;
    std::vector<HealthPickup> healthPickups;
    bool levelCompleted = false;
    int currentLevel = 0;
    parseMap(levelGenerator.getLevel(currentLevel), cellSize, world, player, walls, pits, enemies,
        exit, healthPickups, traps, keys, doors);
    for (int i = 0; i < player.lives; ++i) {
        Heart heart;
        heart.shape = sf::CircleShape(8.0f, 30);
        heart.shape.setFillColor(sf::Color::Red);
        heart.shape.setOrigin(sf::Vector2f(8.0f, 8.0f));
        heart.shape.setPosition(sf::Vector2f(20.0f + i * 21.0f, 20.0f));
        hearts.push_back(heart);
    }
    ContactListener* contactListener = new ContactListener(
        bullets, enemies, player, exit, healthPickups,
        levelCompleted, world, traps, keys, doors,
        playerDeaths, enemiesKilled, trapsTriggered, healthPicked
    );
    world.SetContactListener(contactListener);

    auto lastShotTime = std::chrono::steady_clock::now();
    sf::View view(sf::Vector2f(400.f, 300.f), sf::Vector2f(800.f, 600.f));
    sf::View uiView = window.getDefaultView();
    sf::Clock clock;
    while (window.isOpen()) {
        float deltaTime = clock.restart().asSeconds();
        if (player.enemyStartDelayTimer > 0) {
            player.enemyStartDelayTimer -= deltaTime;
            if (player.enemyStartDelayTimer <= 0) {
                player.enemiesCanMove = true;
                std::cout << "Enemies can now move!" << std::endl;
            }
        }
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            if (event->is<sf::Event::KeyPressed>() &&
                event->getIf<sf::Event::KeyPressed>()->code == sf::Keyboard::Key::Escape) {
                window.close();
            }
        }
        updatePlayer(player, deltaTime);
        view.setCenter(player.shape.getPosition());

        // Стрельба
        auto currentTime = std::chrono::steady_clock::now();
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space) &&
            player.lastShotTime >= 0.5f) {
            createBullet(bullets, player, world);
            player.lastShotTime = 0.0f;
        }
        world.Step(1.0f / 60.0f, 8, 3);
        if (player.enemiesCanMove) {
            updateEnemies(enemies, player.shape.getPosition(), world, deltaTime, doors);
        }
        else {
            for (auto* enemy : enemies) {
                if (enemy->body) {
                    enemy->shape.setPosition(sf::Vector2f(enemy->body->GetPosition().x, enemy->body->GetPosition().y));
                }
            }
        }
        updateBullets(bullets, world);

        updateHealthPickups(healthPickups, world);

        updateTraps(traps, world, deltaTime);

        updateDoors(doors, world);

        updateHealthPickupsAnimation(healthPickups, deltaTime);

        updateHearts(hearts, player, player.bonusLives);
        if (levelCompleted) {
            std::cout << "Level " << currentLevel + 1 << " passed! Good job!" << std::endl;
            currentLevel++;
            auto levelEndTime = std::chrono::steady_clock::now();
            float levelTime = std::chrono::duration<float>(levelEndTime - levelStartTime).count();

            levelGenerator.updateDifficulty(levelTime, playerDeaths, enemiesKilled,
                trapsTriggered, healthPicked);
            levelGenerator.endLevelEvaluation(currentLevel, levelTime,
                playerDeaths, enemiesKilled,
                trapsTriggered, healthPicked);

            levelGenerator.updateDifficulty(levelTime, playerDeaths, enemiesKilled,
                trapsTriggered, healthPicked);
            playerDeaths = 0;
            enemiesKilled = 0;
            trapsTriggered = 0;
            healthPicked = 0;

            if (player.lives <= 0) {
                std::cout << "Game Over! Restarting..." << std::endl;

                clearGameObjects(world, walls, pits, enemies, bullets,
                    healthPickups, traps, keys, doors);

                player = Player();
                player.shape = sf::CircleShape(12.0f, 30);
                player.shape.setFillColor(sf::Color::Blue);
                player.shape.setOrigin(sf::Vector2f(12.0f, 12.0f));
                player.lives = 3;
                player.speed = 100.0f;
                player.angle = 0.0f;
                player.keys = 0;
                player.bonusLives = 0;

                player.directionArc.setPointCount(3);
                player.directionArc.setPoint(0, sf::Vector2f(0.0f, -16.0f));
                player.directionArc.setPoint(1, sf::Vector2f(8.0f, 0.0f));
                player.directionArc.setPoint(2, sf::Vector2f(-8.0f, 0.0f));
                player.directionArc.setFillColor(sf::Color(255, 0, 0, 150));
                player.directionArc.setOrigin(sf::Vector2f(0.0f, 0.0f));

                b2BodyDef playerDef;
                playerDef.type = b2_dynamicBody;
                playerDef.position.Set(400.0f, 300.0f);
                player.body = world.CreateBody(&playerDef);

                b2CircleShape playerShape;
                playerShape.m_radius = 12.0f;

                b2FixtureDef playerFixture;
                playerFixture.shape = &playerShape;
                playerFixture.density = 1.0f;
                playerFixture.filter.categoryBits = PLAYER_CATEGORY;
                playerFixture.filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY |
                    KEY_CATEGORY | HEALTH_CATEGORY | TRAP_CATEGORY | EXIT_CATEGORY;
                player.body->CreateFixture(&playerFixture);

                resetContactListener(world, contactListener, bullets, enemies, player, exit,
                    healthPickups, levelCompleted, traps, keys, doors,
                    playerDeaths, enemiesKilled, trapsTriggered, healthPicked);

                currentLevel = 0;
                parseMap(levelGenerator.getLevel(currentLevel), cellSize, world, player, walls, pits,
                    enemies, exit, healthPickups, traps, keys, doors);

                levelCompleted = false;
                updateHearts(hearts, player, player.bonusLives);
                continue;
            }
            else if (currentLevel < levelGenerator.getLevelCount()) {
                clearGameObjects(world, walls, pits, enemies, bullets,
                    healthPickups, traps, keys, doors);

                int savedLives = player.lives;
                int savedBonusLives = player.bonusLives;
                int savedKeys = player.keys;
                float savedSpeed = player.speed;
                float savedAngle = player.angle;

                player = Player();
                player.lives = savedLives;
                player.bonusLives = savedBonusLives;
                player.keys = savedKeys;
                player.speed = savedSpeed;
                player.angle = savedAngle;
                player.shape = sf::CircleShape(12.0f, 30);
                player.shape.setFillColor(sf::Color::Blue);
                player.shape.setOrigin(sf::Vector2f(12.0f, 12.0f));

                player.directionArc.setPointCount(3);
                player.directionArc.setPoint(0, sf::Vector2f(0.0f, -16.0f));
                player.directionArc.setPoint(1, sf::Vector2f(8.0f, 0.0f));
                player.directionArc.setPoint(2, sf::Vector2f(-8.0f, 0.0f));
                player.directionArc.setFillColor(sf::Color(255, 0, 0, 150));
                player.directionArc.setOrigin(sf::Vector2f(0.0f, 0.0f));
                b2BodyDef playerDef;
                playerDef.type = b2_dynamicBody;
                playerDef.position.Set(400.0f, 300.0f);
                player.body = world.CreateBody(&playerDef);

                b2CircleShape playerShape;
                playerShape.m_radius = 12.0f;

                b2FixtureDef playerFixture;
                playerFixture.shape = &playerShape;
                playerFixture.density = 1.0f;
                playerFixture.filter.categoryBits = PLAYER_CATEGORY;
                playerFixture.filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY; 
                player.body->CreateFixture(&playerFixture);

                resetContactListener(world, contactListener, bullets, enemies, player, exit,
                    healthPickups, levelCompleted, traps, keys, doors,
                    playerDeaths, enemiesKilled, trapsTriggered, healthPicked);

                parseMap(levelGenerator.getLevel(currentLevel), cellSize, world, player, walls, pits,
                    enemies, exit, healthPickups, traps, keys, doors);

                levelCompleted = false;

                updateHearts(hearts, player, player.bonusLives);

                continue;
            }
            else {
                std::cout << "You passed all levels, congratulations! Restarting..." << std::endl;

                clearGameObjects(world, walls, pits, enemies, bullets,
                    healthPickups, traps, keys, doors);
                player = Player();
                player.shape = sf::CircleShape(12.0f, 30);
                player.shape.setFillColor(sf::Color::Blue);
                player.shape.setOrigin(sf::Vector2f(12.0f, 12.0f));
                player.lives = 3;
                player.speed = 100.0f;
                player.angle = 0.0f;
                player.keys = 0;
                player.bonusLives = 0;

                player.directionArc.setPointCount(3);
                player.directionArc.setPoint(0, sf::Vector2f(0.0f, -16.0f));
                player.directionArc.setPoint(1, sf::Vector2f(8.0f, 0.0f));
                player.directionArc.setPoint(2, sf::Vector2f(-8.0f, 0.0f));
                player.directionArc.setFillColor(sf::Color(255, 0, 0, 150));
                player.directionArc.setOrigin(sf::Vector2f(0.0f, 0.0f));

                b2BodyDef playerDef;
                playerDef.type = b2_dynamicBody;
                playerDef.position.Set(400.0f, 300.0f);
                player.body = world.CreateBody(&playerDef);

                b2CircleShape playerShape;
                playerShape.m_radius = 12.0f;

                b2FixtureDef playerFixture;
                playerFixture.shape = &playerShape;
                playerFixture.density = 1.0f;
                playerFixture.filter.categoryBits = PLAYER_CATEGORY;
                playerFixture.filter.maskBits = ENEMY_CATEGORY | WALL_CATEGORY | DOOR_CATEGORY;
                player.body->CreateFixture(&playerFixture);

                resetContactListener(world, contactListener, bullets, enemies, player, exit,
                    healthPickups, levelCompleted, traps, keys, doors,
                    playerDeaths, enemiesKilled, trapsTriggered, healthPicked);
                currentLevel = 0;
                parseMap(levelGenerator.getLevel(currentLevel), cellSize, world, player, walls, pits,
                    enemies, exit, healthPickups, traps, keys, doors);

                levelCompleted = false;
                updateHearts(hearts, player, player.bonusLives);
                continue;
            }
        }

        window.clear();
        window.setView(view);
        window.draw(exit.shape);
        for (const auto& health : healthPickups) {
            if (health.active) {
                window.draw(health.shape);
            }
        }

        for (const auto& wall : walls) {
            window.draw(wall.shape);
        }

        for (const auto& trap : traps) {
            if (trap.active) {
                window.draw(trap.shape);
            }
        }


        for (const auto& pit : pits) {
            window.draw(pit.shape);
        }

        for (const auto& key : keys) {
            if (!key.collected) {
                window.draw(key.shape);
            }
        }

        for (const auto& door : doors) {
            if (!door.opened) {
                window.draw(door.shape);
            }
        }
        window.draw(player.shape);
        window.draw(player.directionArc);

        for (const auto* enemy : enemies) {
            if (enemy->health > 0) {
                window.draw(enemy->shape);
            }
        }
        for (const auto& bullet : bullets) {
            window.draw(bullet->shape);
        }

        window.setView(uiView);
        for (const auto& heart : hearts) {
            window.draw(heart.shape);
        }
        drawKeys(window, player); 

        window.display();
    }
   
    for (auto* bullet : bullets) {
        world.DestroyBody(bullet->body);
        delete bullet;
    }
    for (auto* enemy : enemies) {
        world.DestroyBody(enemy->body);
        delete enemy;
    }

    delete contactListener;
    levelGenerator.saveState();
    return 0;
}