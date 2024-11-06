# genetic_algorithm.py

import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from tqdm.notebook import tqdm
from scipy.spatial.distance import pdist

class GeneticAlgorithmCV:
    def __init__(
        self,
        model_type,
        estimator,
        param_grid,
        cv=None,
        pop_size=25,
        generations=15,
        early_stopping_rounds=3,
        crossover_initial=0.1,
        crossover_end=0.9,
        mutation_initial=0.1,
        mutation_end=0.9,
        elitism=True,
        elite_size=2,
        tournament_size=3,
        n_random=5,
        n_jobs=-1,
        verbose=False,
        alpha=1.0,
        beta=1.0,
    ):
        self.model_type = model_type
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.pop_size = pop_size
        self.generations = generations
        self.early_stopping_rounds = early_stopping_rounds
        self.crossover_initial = crossover_initial
        self.crossover_end = crossover_end
        self.mutation_initial = mutation_initial
        self.mutation_end = mutation_end
        self.elitism = elitism
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.n_random = n_random
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.alpha = alpha
        self.beta = beta
        self.fitness_history = []
        self.diversity_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def decode_chromosome(self, chromosome):
        param_values = {}
        for i, key in enumerate(self.param_grid.keys()):
            gene = chromosome[i]
            param_info = self.param_grid[key]
            low = param_info['low']
            high = param_info['high']
            if param_info['type'] == 'int':
                value = int(np.round(gene * (high - low) + low))
            elif param_info['type'] == 'float':
                value = gene * (high - low) + low
            param_values[key] = value
        return param_values

    def initialize_population(self):
        chromosome_length = len(self.param_grid)
        population = np.random.uniform(low=0.0, high=1.0, size=(self.pop_size, chromosome_length))
        return population

    def evaluate_population(self, population, X_train, y_train):
        def evaluate_individual(chromosome):
            params = self.decode_chromosome(chromosome)
            scores_f1 = []
            scores_auc = []
            for train_idx, val_idx in self.cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                model = clone(self.estimator)
                model.set_params(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                y_pred_prob = model.predict_proba(X_val)[:, 1]
                score_f1 = f1_score(y_val, y_pred, average='binary')
                score_auc = roc_auc_score(y_val, y_pred_prob)
                scores_f1.append(score_f1)
                scores_auc.append(score_auc)
            fitness_f1 = np.mean(scores_f1)
            fitness_auc = np.mean(scores_auc)
            fitness = (fitness_f1 + fitness_auc) / 2
            return fitness

        if self.n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        fitnesses = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_individual)(chromosome) for chromosome in population
        )
        return np.array(fitnesses)

    def select_parents(self, population, fitnesses):
        selected = []
        if self.elitism:
            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = population[sorted_indices[:self.elite_size]]
            selected.extend(elites.tolist())
            non_elite_indices = sorted_indices[self.elite_size:]
            non_elite_population = population[non_elite_indices]
            non_elite_fitnesses = fitnesses[non_elite_indices]
        else:
            non_elite_population = population
            non_elite_fitnesses = fitnesses
        num_parents_needed = self.pop_size - (self.elite_size if self.elitism else 0)
        for _ in range(num_parents_needed):
            if len(non_elite_population) < self.tournament_size:
                current_tournament_size = len(non_elite_population)
            else:
                current_tournament_size = self.tournament_size
            tournament_indices = np.random.choice(len(non_elite_population), size=current_tournament_size, replace=False)
            tournament_fitnesses = non_elite_fitnesses[tournament_indices]
            best_tournament_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(non_elite_population[best_tournament_idx])
        selected = np.vstack(selected)
        return selected

    def crossover(self, parents, crossover_rate):
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i].copy()
            parent2 = parents[(i+1) % len(parents)].copy()
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, len(parent1))
                child1 = np.concatenate((parent1[:point], parent2[point:]))
                child2 = np.concatenate((parent2[:point], parent1[point:]))
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1)
                offspring.append(parent2)
        if len(parents) % 2 != 0:
            offspring.append(parents[-1])
        return np.vstack(offspring)

    def mutate(self, offspring, mutation_rate, mutation_scale=0.1):
        for chromosome in offspring:
            for gene_idx in range(len(chromosome)):
                if np.random.rand() < mutation_rate:
                    mutation = np.random.normal(0, mutation_scale)
                    chromosome[gene_idx] += mutation
                    chromosome[gene_idx] = np.clip(chromosome[gene_idx], 0.0, 1.0)
        return offspring

    def generate_random_individuals(self, n_random):
        chromosome_length = len(self.param_grid)
        random_chromosomes = np.empty((n_random, chromosome_length), dtype=np.float32)
        for i, key in enumerate(self.param_grid.keys()):
            grid = self.param_grid[key]
            low = grid['low']
            high = grid['high']
            if grid['type'] == 'int':
                sampled = np.random.randint(low, high + 1, size=n_random)
                normalized = (sampled - low) / (high - low)
                random_chromosomes[:, i] = normalized.astype(np.float32)
            elif grid['type'] == 'float':
                sampled = np.random.uniform(low, high, size=n_random)
                normalized = (sampled - low) / (high - low)
                random_chromosomes[:, i] = normalized.astype(np.float32)
            else:
                raise ValueError(f"Tipo de parámetro no soportado: {grid['type']}")
        return random_chromosomes
    
    def calculate_diversity(self, population):
        if len(population) <= 1:
            return 0.0
        distances = pdist(population, metric='euclidean')
        mean_distance = np.mean(distances)
        return mean_distance
    
    def normalize_metric(self, history):
        min_val = np.min(history)
        max_val = np.max(history)
        if max_val - min_val == 0:
            return 0.0
        else:
            return (history[-1] - min_val) / (max_val - min_val)

    def fit(self, X_train, y_train):
        population = self.initialize_population()
        best_overall_fitness = -np.inf
        best_overall_chromosome = None
        no_improvement_generations = 0

        for generation in tqdm(range(self.generations), desc=f"Generaciones {self.model_type}", unit="gen"):
            # Evaluar la población
            fitnesses = self.evaluate_population(population, X_train, y_train)
            # Capturar máximo y mínimo
            current_best_fitness = np.max(fitnesses)
            # Añadir el mejor fitness al historial
            self.fitness_history.append(current_best_fitness)
            # Calcular mejora en el fitness
            if len(self.fitness_history) > 1:
                fitness_improvement = self.fitness_history[-1] - self.fitness_history[-2]
            else:
                fitness_improvement = 0.0
            # Calcular diversidad
            diversity = self.calculate_diversity(population)
            self.diversity_history.append(diversity)
            # Normalizar métricas
            fitness_improvement_norm = self.normalize_metric(self.fitness_history)
            diversity_norm = self.normalize_metric(self.diversity_history)
            # Combinar métricas y aplicar sigmoide
            combined_metric = self.alpha * fitness_improvement_norm + self.beta * diversity_norm
            sigmoid_value = self.sigmoid(combined_metric)
            # Ajustar tasas de cruce y mutación
            # Ajustar tasas de cruce y mutación
            crossover_rate = self.crossover_initial + (self.crossover_end - self.crossover_initial) * sigmoid_value
            mutation_rate = self.mutation_initial + (self.mutation_end - self.mutation_initial) * (1 - sigmoid_value)
            # Asegurar que las tasas estén dentro de [0,1]
            crossover_rate = np.clip(crossover_rate, 0.0, 1.0)
            mutation_rate = np.clip(mutation_rate, 0.0, 1.0)
            # Actualizar el mejor fitness y cromosoma
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_idx = np.argmax(fitnesses)
                best_overall_chromosome = population[best_idx]
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            if self.verbose:
                print(f"[{generation+1}, {self.model_type}] Fitness: {current_best_fitness} | Best Fitness: {best_overall_fitness}")
                print(f"[{generation+1}, {self.model_type}] Fitness Improvement: {fitness_improvement:.4f} | Diversity: {diversity:.4f}")
                print(f"[{generation+1}, {self.model_type}] Normalized Fitness Improvement: {fitness_improvement_norm:.4f} | Normalized Diversity: {diversity_norm:.4f}")
                print(f"[{generation+1}, {self.model_type}] Crossover Rate: {crossover_rate:.4f} | Mutation Rate: {mutation_rate:.4f}")

            # Verificar condición de parada por falta de mejora
            if no_improvement_generations >= self.early_stopping_rounds:
                if self.verbose:
                    print(f"[{generation+1}, {self.model_type}] Early stopping due to no improvement.")
                    print(f"Best fitness for {self.model_type}: {best_overall_fitness}")
                break

            # Seleccionar padres
            parents = self.select_parents(population, fitnesses)
            # Generar descendencia mediante cruza
            offspring = self.crossover(parents, crossover_rate=crossover_rate)
            # Aplicar mutaciones a la descendencia
            offspring = self.mutate(offspring, mutation_rate=mutation_rate)
            # Ajustar el tamaño final de la población a pop_size
            if len(offspring) > self.pop_size:
                offspring = offspring[:self.pop_size]
            # Inyección de individuos aleatorios
            if self.n_random > 0:
                random_individuals = self.generate_random_individuals(self.n_random)
                offspring = np.vstack((offspring, random_individuals))
            # Actualizar la población para la siguiente generación
            population = offspring

        if best_overall_chromosome is not None:
            self.best_score_ = best_overall_fitness
            self.best_params_ = self.decode_chromosome(best_overall_chromosome)
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X_train, y_train)
        else:
            if self.verbose:
                print("No se encontró un cromosoma válido durante el entrenamiento.")
            self.best_params_ = None
            self.best_estimator_ = None