# genetic_algorithm.py
import os
import tempfile
import numpy as np
from memory_profiler import profile
from joblib import Parallel, delayed, parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from scipy.spatial.distance import pdist

class GeneticAlgorithmCV:
    def __init__(
        self,
        model_type,
        pipeline,
        param_grid,
        estimator_map=None,
        cv=None,
        pop_size=25,
        generations=5,
        early_stopping_rounds=2,
        crossover_initial=0.9,
        crossover_end=0.1,
        mutation_initial=0.1,
        mutation_end=0.9,
        elitism=True,
        elite_size=3,
        tournament_size=3,
        n_random=5,
        n_jobs=-1,
        verbose=False,
        alpha=1.0,
        beta=1.0
    ):
        self.model_type = model_type
        self.pipeline = pipeline
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
        self.best_params_full_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.alpha = alpha
        self.beta = beta
        self.fitness_history = []
        self.diversity_history = []
        if estimator_map is None:
            estimator_map = {}
            self.estimator_map = estimator_map
            self.estimator_map = estimator_map
        else:
            self.estimator_map = estimator_map

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def decode_chromosome(self, chromosome):
        param_values = {}
        param_estimators = {}
        skip_steps = set()
        for i, key in enumerate(self.param_grid.keys()):
            gene = chromosome[i]
            param_info = self.param_grid[key]
            if param_info['type'] == 'int':
                low = param_info['low']
                high = param_info['high']
                value = int(np.round(gene * (high - low) + low))
                param_values[key] = value
            elif param_info['type'] == 'float':
                low = param_info['low']
                high = param_info['high']
                value = gene * (high - low) + low
                param_values[key] = value
            elif param_info['type'] == 'categorical':
                categories = param_info['values']
                k = len(categories)
                index = int(np.floor(gene * k))
                if index >= k:
                    index = k - 1
                value = categories[index]
                param_values[key] = value
                if key in self.estimator_map:
                    estimator = self.estimator_map[key][value]
                    param_estimators[key] = estimator
                    if estimator == 'passthrough':
                        skip_steps.add(key)
            else:
                raise ValueError(f"Tipo de parámetro no soportado: {param_info['type']}")
        for step in skip_steps:
            keys_to_remove = [k for k in list(param_values.keys()) if k.startswith(f"{step}__")]
            for k in keys_to_remove:
                del param_values[k]
        for key in list(param_values.keys()):
            if '__' in key:
                step_name, param_name = key.split('__', 1)
                if step_name in param_estimators:
                    estimator = param_estimators[step_name]
                    if estimator == 'passthrough':
                        continue
                else:
                    estimator = dict(self.pipeline.named_steps)[step_name]
        return param_values, param_estimators

    def initialize_population(self):
        chromosome_length = len(self.param_grid)
        population = np.random.uniform(low=0.0, high=1.0, size=(self.pop_size, chromosome_length))
        return population

    def evaluate_population(self, population, X_train_memmap, y_train_memmap):
        #@profile
        def evaluate_individual(chromosome):
            try:
                param_values, param_estimators = self.decode_chromosome(chromosome)
                params = param_values.copy()
                params.update(param_estimators)
                model = clone(self.pipeline)
                model.set_params(**params)
                scores_acur = []
                for train_idx, val_idx in self.cv.split(X_train_memmap, y_train_memmap):
                    model.fit(X_train_memmap[train_idx], y_train_memmap[train_idx])
                    y_pred = model.predict(X_train_memmap[val_idx])
                    score_acur = accuracy_score(y_train_memmap[val_idx], y_pred)
                    scores_acur.append(score_acur)
                fitness_acur = np.mean(scores_acur)
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluando el individuo: {e}")
                return -np.inf
            return fitness_acur

        with parallel_backend("loky", inner_max_num_threads=1):
            fitnesses = Parallel(n_jobs=self.n_jobs)(
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
        num_parents = len(parents)
        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i].copy()
            parent2 = parents[i + 1].copy()
            if np.random.rand() < crossover_rate:
                mask = np.random.rand(len(parent1)) < 0.5
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1)
                offspring.append(parent2)
        if num_parents % 2 != 0:
            offspring.append(parents[-1])
        return np.vstack(offspring)

    def mutate(self, offspring, mutation_rate, mutation_scale=0.1):
        for chromosome in offspring:
            for gene_idx, key in enumerate(self.param_grid.keys()):
                if np.random.rand() < mutation_rate:
                    param_info = self.param_grid[key]
                    if param_info['type'] in ['int', 'float']:
                        mutation = np.random.normal(0, mutation_scale)
                        chromosome[gene_idx] += mutation
                        chromosome[gene_idx] = np.clip(chromosome[gene_idx], 0.0, 1.0)
                    elif param_info['type'] == 'categorical':
                        categories = param_info['values']
                        k = len(categories)
                        current_gene_value = chromosome[gene_idx]
                        current_index = int(np.floor(current_gene_value * k))
                        if current_index >= k:
                            current_index = k - 1
                        possible_indices = list(range(k))
                        possible_indices.remove(current_index)
                        if possible_indices:
                            new_index = np.random.choice(possible_indices)
                            if k > 1:
                                chromosome[gene_idx] = new_index / (k - 1)
                            else:
                                chromosome[gene_idx] = 0.0
                    else:
                        raise ValueError(f"Tipo de parámetro no soportado: {param_info['type']}")
        return offspring

    def generate_random_individuals(self, n_random):
        chromosome_length = len(self.param_grid)
        random_chromosomes = np.empty((n_random, chromosome_length), dtype=np.float32)
        for i, key in enumerate(self.param_grid.keys()):
            grid = self.param_grid[key]
            if grid['type'] == 'int':
                low = grid['low']
                high = grid['high']
                sampled = np.random.randint(low, high + 1, size=n_random)
                normalized = (sampled - low) / (high - low)
                random_chromosomes[:, i] = normalized.astype(np.float32)
            elif grid['type'] == 'float':
                low = grid['low']
                high = grid['high']
                sampled = np.random.uniform(low, high, size=n_random)
                normalized = (sampled - low) / (high - low)
                random_chromosomes[:, i] = normalized.astype(np.float32)
            elif grid['type'] == 'categorical':
                categories = grid['values']
                k = len(categories)
                sampled_indices = np.random.randint(0, k, size=n_random)
                if k > 1:
                    normalized = sampled_indices / (k - 1)
                else:
                    normalized = np.zeros(n_random)
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
        # Crear archivos temporales utilizando NamedTemporaryFile
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as X_temp_file:
            # Guardar X_train en el archivo temporal
            np.save(X_temp_file, X_train)
            X_temp_path = X_temp_file.name  # Obtener el nombre del archivo temporal

        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as y_temp_file:
            # Guardar y_train en el archivo temporal
            np.save(y_temp_file, y_train)
            y_temp_path = y_temp_file.name  # Obtener el nombre del archivo temporal

        # Cargar los datos como memmap
        X_train_memmap = np.load(X_temp_path, mmap_mode='r')
        y_train_memmap = np.load(y_temp_path, mmap_mode='r')

        # Ahora, utiliza X_train_memmap e y_train_memmap en lugar de X_train e y_train
        population = self.initialize_population()
        best_overall_fitness = -np.inf
        best_overall_chromosome = None
        no_improvement_generations = 0

        for generation in tqdm(range(self.generations), desc=f"Generaciones {self.model_type}", unit="gen"):
            # Evaluar la población
            fitnesses = self.evaluate_population(population, X_train_memmap, y_train_memmap)
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
            if(fitness_improvement_norm != 0 or diversity_norm != 0):
                # Combinar métricas y aplicar sigmoide
                combined_metric = self.alpha * fitness_improvement_norm + self.beta * diversity_norm
                sigmoid_value = self.sigmoid(combined_metric)
                # Ajustar tasas de cruce y mutación
                crossover_rate = self.crossover_initial + (self.crossover_end - self.crossover_initial) * sigmoid_value
                mutation_rate = self.mutation_initial + (self.mutation_end - self.mutation_initial) * sigmoid_value
            else:
                crossover_rate = self.crossover_initial * ((self.crossover_end / self.crossover_initial) ** (generation / self.generations))
                mutation_rate = self.mutation_initial * ((self.mutation_end / self.mutation_initial) ** (generation / self.generations))
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
            offspring = self.crossover(
                parents,
                crossover_rate=crossover_rate
            )
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

        # Eliminar los archivos temporales
        os.unlink(X_temp_path)
        os.unlink(y_temp_path)
        # Devolver el mejor pipeline
        if best_overall_chromosome is not None:
            self.best_score_ = best_overall_fitness
            self.best_params_, self.best_estimators_ = self.decode_chromosome(best_overall_chromosome)
            self.best_params_full_ = self.best_params_.copy()  # Guardamos una copia antes de modificar
            # Reconstruir el pipeline excluyendo pasos en 'passthrough'
            new_steps = []
            for name, step in self.pipeline.steps:
                # Verificar si el paso está configurado como 'passthrough'
                if name in self.best_estimators_ and self.best_estimators_[name] == 'passthrough':
                    continue  # Omitir este paso
                else:
                    # Si el estimador del paso ha sido especificado en best_estimators_
                    if name in self.best_estimators_:
                        # Usar el estimador seleccionado
                        step = self.best_estimators_[name]
                    else:
                        # Clonar el paso original para preservar los parámetros predeterminados
                        step = clone(step)
                    # Obtener los parámetros específicos para este paso que están en param_grid
                    optimized_params = {
                        key.split('__', 1)[1]: value
                        for key, value in self.best_params_.items()
                        if key.startswith(f"{name}__")
                    }
                    # Remover estos parámetros de best_params_ para evitar duplicados
                    for key in list(self.best_params_.keys()):
                        if key.startswith(f"{name}__"):
                            del self.best_params_[key]
                    # Actualizar los parámetros del paso si es necesario
                    if optimized_params:
                        step.set_params(**optimized_params)
                    new_steps.append((name, step))
            # Crear el nuevo pipeline con los pasos actualizados
            self.best_estimator_ = Pipeline(new_steps)
            self.best_estimator_.fit(X_train, y_train)
        else:
            if self.verbose:
                print("No se encontró un cromosoma válido durante el entrenamiento.")
            self.best_params_ = None
            self.best_estimator_ = None