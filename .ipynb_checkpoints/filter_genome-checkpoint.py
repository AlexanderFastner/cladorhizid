#!bin/bash
#-------------------------------------------------------------------------------------------------------------------------------
#DESCRIPTION
#1 get GC content from contigs and in chunk sizes, then visualize both
#2 get coverage for contigs and in chunk sizes, then visualize both
#3 histogram of contig lengths
#4 blob plot (gc vs average coverage) for both
#5 kmeans clustering
#6 take cluster from kmeans as input to blast
#7 blob plot again and mark points that had accurate blast hits
#8 histogram for length distribution
#9 take contigs starting with largest up til genome size
#10 add contig ids to fasta file
#-------------------------------------------------------------------------------------------------------------------------------
#IMPORTS
import typer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import shutil
import csv
import multiprocess
from typing import List, Optional
from typing_extensions import Annotated
from sklearn.cluster import KMeans
from scipy import stats
#-------------------------------------------------------------------------------------------------------------------------------
#PARAMETERS
# assembly : The fasta input assembly file
# coverage_per_contig : A tsv file with information for each contig
# coverage_per_base:  A tsv file with coverage information for each base pair position
# n_cores: number of cpu cores to multiprocess with
# out_dir : output directory
# min_contig_size : The minimum size of a contig, otherwise its ignored
# chunk_size : The chunk size for dividing up longer contigs (default 10000)
# visualize_gc : An option for creating plots showing the distribution of GC content accross a contig in steps of a given chunk size (default = True)
# visualize_coverage : An option to create plots showing coverage distribution per contig
# visualize_histogram : Option to cisualize histogram of conitg lengths
# threshold_pct : The percentage away from the mean that something needs to be in order to be highlighted
# visualize_blob : Option to visualize blob plot
# plot_min_length : The minimum contig length to be plotted in the Blob plot
# n_clusters : Number of clusters for kmeans
# len_to_blast : Length of the to be blasted subsections
# space_to_blast : Distance between subsections to be blasted
#-------------------------------------------------------------------------------------------------------------------------------
app = typer.Typer()

#TODO add this
#coverage_per_contig = "./data/heliopora_coerulea_hifi_2_p_assembly.coverage"
#coverage_per_base = "./data/heliopora_coerulea_hifi_2_p_assembly.depth"

@app.command()
def main(assembly: str = '',
         coverage_per_contig: str = '',
         coverage_per_base: str = '',
         n_cores: int = 4,
         out_dir: Annotated[str, typer.Argument()] = './out_dir',
         min_contig_size: Annotated[int, typer.Argument()] = 100000,
         chunk_size: Annotated[int, typer.Argument()] = 10000,
         visualize_gc: Annotated[bool, typer.Argument()] = True,
         visualize_coverage: Annotated[bool, typer.Argument()] = True,
         visualize_histogram: Annotated[bool, typer.Argument()] = True,
         threshold_pct: Annotated[int, typer.Argument()] = 30,
         visualize_blob: Annotated[bool, typer.Argument()] = True,
         plot_min_length: Annotated[int, typer.Argument()] = 100000,
         n_clusters: Annotated[int, typer.Argument()] = 2,
         len_to_blast: Annotated[int, typer.Argument()] = 1000,
         space_to_blast: Annotated[int, typer.Argument()] = 10000,
         threshold: Annotated[float, typer.Argument()] = 0.8
    ):
    """
    Program to analyze long (Pacbio) contigs then filter based on gc content, coverage, and blast hits
    """

    #---------------------------------------------------------------------------------------------------------------------------
    # Naive GC content for each content
    #---------------------------------------------------------------------------------------------------------------------------

    def get_gc_content(fasta_file):
        """
        Reads in a FASTA file and returns a dictionary mapping contig IDs to their GC content
        """
        gc_content = {}
        current_id = None
        current_seq = ""

        with open(fasta_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    # If this is a new contig, calculate the GC content for the previous one (if there was one)
                    if current_id is not None:
                        #remove if current_seq y min_contig_size
                        if len(current_seq) > min_contig_size:
                            gc_content[current_id] = (current_seq.count("G") + current_seq.count("C")) / len(current_seq)

                    # Start the new contig
                    #current_id = line.strip()[1:].split("/")[3]
                    current_id = line.strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()

        # Calculate GC content for the final contig
        gc_content[current_id] = (current_seq.count("G") + current_seq.count("C")) / len(current_seq)

        return gc_content

    print('Starting Naive GC content')
    naive_gc = get_gc_content(assembly)
    #print(naive_gc)
    #print(len(naive_gc))
    print('Finished Naive GC content')

    #---------------------------------------------------------------------------------------------------------------------------
    # GC content for each contig by averaging in given chunk sizes
    #---------------------------------------------------------------------------------------------------------------------------

    def get_gc_content_chunks(fasta_file, chunk_size):
        """
        Reads in a FASTA file and returns a dictionary mapping contig IDs to a list of gc content per chunk size
        """
        gc_content = {}
        current_id = None
        current_seq = ""
        chunks = []

        with open(fasta_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    # If this is a new contig, calculate the GC content for the previous one (if there was one)
                    if current_id is not None:
                        #remove if current_seq < min_contig_size
                        if len(current_seq) > min_contig_size:
                            i = 0
                            while i < len(current_seq):
                                # check for last chunk
                                if (i + chunk_size) > len(current_seq):
                                    end = len(current_seq)
                                else:
                                    end = i + chunk_size
                                sub = current_seq[i:end]
                                gc = round(((sub.count("G") + sub.count("C")) / len(sub)),2)
                                chunks.append(gc)
                                i = end
                            # add to dictionary in form {current_id : [array of GC content of given chunk size]}
                            gc_content[current_id] = [chunks, len(current_seq)]

                    # Start the new contig
                    #current_id = line.strip()[1:].split("/")[3]
                    current_id = line.strip()[1:]
                    current_seq = ""
                    chunks = []

                else:
                    current_seq += line.strip()

            # calculate GC content for the last contig
            if current_id is not None:
                i = 0
                while i < len(current_seq):
                    # check for last chunk
                    if (i + chunk_size) > len(current_seq):
                        end = len(current_seq)
                    else:
                        end = i + chunk_size
                    sub = current_seq[i:end]
                    gc = round(((sub.count("G") + sub.count("C")) / len(sub)), 2)
                    chunks.append(gc)
                    i = end
                # add to dictionary in form {current_id : [array of GC content of given chunk size, length]}
                gc_content[current_id] = [chunks, len(current_seq)]

        return gc_content

    print('Starting chunk GC content')
    chunks = get_gc_content_chunks(assembly, chunk_size)
    gc_trimmed_mean = {}
    #Get the trimmed mean from chunks
    for entry in chunks:
        length = chunks.get(entry)[1]
        avg_cov = chunks.get(entry)[0]
        gc_trimmed_mean[entry] = [stats.trim_mean(avg_cov, 0.05), length]
    #print(gc_trimmed_mean)
    print('Finished chunk GC content')
    print('total length',len(chunks.items()))

    #---------------------------------------------------------------------------------------------------------------------------
    # Visualize GC content
    #---------------------------------------------------------------------------------------------------------------------------

    if visualize_gc:

        save_dir = out_dir + "/GC_visual_output"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        def process_gc_plot(chunk):
            #format id
            id = chunk
            gc_content = np.array(chunks.get(chunk)[0])
            #print(len(gc_content))
            #print(gc_content)

            # create a bar plot
            fig, ax = plt.subplots()
            mean_gc = np.mean(gc_content)
            colors = ['red' if abs(gc - mean_gc) / mean_gc * 100 >= threshold_pct else 'blue' for gc in gc_content]
            plt.bar(range(len(gc_content)), gc_content, color=colors)
            ax.axhline(y=gc_trimmed_mean[id][0], color='black')
            plt.title("Contig: " + str(id))
            plt.xlabel('index of Chunk of size: ' + str(chunk_size))
            plt.ylabel('GC content')
            #save plot
            save_path = os.path.join(save_dir, "GC_content_" + str(id) + ".png")
            plt.savefig(save_path)
            plt.close()
        
        if __name__ == '__main__':
            with multiprocess.Pool(processes=n_cores) as pool:
                print('Starting Visualize GC content')
                pool.map(process_gc_plot, chunks.keys())
                print('Finished Visualize GC content')

    else:
        print("GC visualization is turned off.")

    #---------------------------------------------------------------------------------------------------------------------------
    # Naive Coverage
    #---------------------------------------------------------------------------------------------------------------------------

    #TODO test below here when coverage is done being run
    
    def get_coverage_per_contig(coverage_file):
        naive_coverage = {}

        with open(coverage_file, "r") as f:
            next(f)
            for line in f:
                # Split line into columns
                columns = line.strip().split("\t")

                #only add if length > min_contig_size
                if (int(columns[2]) - int(columns[1])) > min_contig_size:
                    # Use the first column as key and the sixth column as value
                    key = columns[0].split("/")[3]
                    value = columns[6]
                    length = int(columns[2]) - int(columns[1])

                    # Add key-value pair to dictionary
                    naive_coverage[key] = [float(value), length]

        f.close()
        return naive_coverage

    print('Starting Naive Coverage')
    naive_coverage = get_coverage_per_contig(coverage_per_contig)
    print(len(naive_coverage))
    print('Finsihed Naive Coverage')

    #---------------------------------------------------------------------------------------------------------------------------
    # Chunk averaged Coverage
    #---------------------------------------------------------------------------------------------------------------------------
    
    def get_coverage_per_contig_by_chunks(depth_file, chunk_size):
        save_dir = out_dir + "Coverage_output/"
        
        # Define a dictionary to store the coverage data for each ID
        id_data = {}
    
        # Read in the TSV file
        with open(depth_file, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
    
            # Iterate over each row in the file
            for row in reader:
                # Extract the ID, index, and coverage from the row
                id = row[0].split("/")[3]
                index = int(row[1])
                coverage = float(row[2])
    
                # Check if this is the first row for this ID
                if id not in id_data:
                    id_data[id] = []
    
                # Find the window index for this row
                window_index = index // chunk_size
    
                # Check if there is already coverage data for this window
                if len(id_data[id]) <= window_index:
                    # Add a new entry for this window
                    id_data[id].append({'window_sum': coverage, 'window_count': 1})
                else:
                    # Add the coverage to the existing window data
                    id_data[id][window_index]['window_sum'] += coverage
                    id_data[id][window_index]['window_count'] += 1            
                    
              
        # Write the output to a file
        with open(save_dir + "chunked_windows_coverage.tsv", 'w') as outfile:
            outfile.write("id\tindex\taverage_coverage\n")
            for id, windows in id_data.items():
                #check total length > min_contig_size
                total_len = 0
                for i, window in enumerate(windows):
                    total_len += window['window_count']
                    if total_len > min_contig_size:
                        for i, window in enumerate(windows):
                            if window['window_count'] > 0:
                                avg_coverage = round((window['window_sum'] / window['window_count']), 2)
                                outfile.write(f"{id}\t{i}\t{avg_coverage}\n")
                            
            outfile.close()
        
        #now get the average coverage for each id
        # Open the input file
        with open(save_dir + "chunked_windows_coverage.tsv", "r") as f:
            next(f)
            # Create a dictionary to store the total coverage and highest index for each id
            id_coverage = {}
            # Read through each line in the file
            for line in f:
                # Split the line into three columns
                id, index, coverage = line.strip().split("\t")
                # Convert the coverage to float
                index = float(index)
                coverage = float(coverage)
                # If the id is not in the dictionary yet, add it with a coverage of 0 and index of -1
                if id not in id_coverage:
                    id_coverage[id] = {"coverage": 0, "max_index": -1}
                # Add the coverage to the total for this id
                id_coverage[id]["coverage"] += coverage
                # Update the max index for this id if the current index is higher
                if index > id_coverage[id]["max_index"]:
                    id_coverage[id]["max_index"] = index
        f.close()
        
        #make list of coverages, then take trimmed mean of that list
        coverage_trimmed_mean = {}
        for entry in id_data:
            coverages = []
            for chunk in id_data.get(entry):
                window_sum = float(chunk.get('window_sum'))
                window_count = float(chunk.get('window_count'))
                coverage = round((window_sum / window_count), 2)
                coverages.append(coverage)
            
            coverage_trimmed_mean[entry] = round(stats.trim_mean(coverages, 0.05), 2)
            
        # Open the output file
        with open(save_dir + "chunked_coverage.tsv", "w") as f:
            # Write the header row
            f.write("id\taverage_coverage\n")
            # Loop through the ids in the dictionary
            for id in coverage_trimmed_mean:
                average_coverage = coverage_trimmed_mean[id]
                f.write("{}\t{}\n".format(id, average_coverage))
        f.close()
    
        return id_data
        

    chunks_coverage = get_coverage_per_contig_by_chunks(coverage_per_base, chunk_size)
    print(len(chunks_coverage))
    
    #---------------------------------------------------------------------------------------------------------------------------
    # Visualize Coverage
    #---------------------------------------------------------------------------------------------------------------------------

    save_dir = out_dir + "Coverage_output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #windows
    data = pd.read_csv(save_dir + "chunked_windows_coverage.tsv", dtype={'id': 'object'}, sep = "\t")
    trimmed_mean = pd.read_csv(save_dir + "chunked_coverage.tsv", dtype={'id': 'object'}, sep = "\t")
    
    # Convert data to pandas dataframe
    df = pd.DataFrame(data)
    mean = pd.DataFrame(trimmed_mean)
    # Group data by id
    groups = df.groupby('id')
    # Get a list of unique ids
    unique_ids = df['id'].unique()
        
    if visualize_Coverage:

        def process_coverage_plots(id):
            # Filter the dataframe to get only the rows for this id
            df_id = df[df['id'] == id]
            
            fig, ax = plt.subplots()
            trimmed_mean_coverage = float(mean.loc[mean['id'] == id, 'average_coverage'])
            ax.axhline(y=trimmed_mean_coverage, color='black')
            #filtered colors
            colors = ['red' if abs(coverage - trimmed_mean_coverage) / trimmed_mean_coverage * 100 >= threshold_pct else 'blue' for coverage in df_id['average_coverage']]
            plt.bar(df_id['index'], df_id['average_coverage'], label=f"id={id}", color=colors)
            plt.xlabel('Index of chunk size ' + str(chunk_size))
            plt.ylabel('Average coverage')
            # save the plot
            save_path = os.path.join(save_dir, "Coverage_" + str(id) + ".png")
            plt.savefig(save_path)
            plt.close()

        if __name__ == '__main__':
            with multiprocess.Pool(processes=n_cores) as pool:
                print('Starting Visualize Coverage')
                pool.map(process_coverage_plots, unique_ids)
                print('Finished Visualize Coverage')
                
    else :
        print("Coverage visualization is turned off.")

    #---------------------------------------------------------------------------------------------------------------------------
    # Blob Naive Plot
    #---------------------------------------------------------------------------------------------------------------------------

    #plot gc content (x) vs coverage (y)
    if visualize_blob:
        
        data = [{'id': k, 'gc_content': v} for k, v in naive_gc.items()]
        data2 = [{'id': k, 'average_coverage': v[0], 'length': v[1]} for k, v in naive_coverage.items()]
        # Convert the list of dictionaries to dataframes
        df1 = pd.DataFrame(data)
        df2 = pd.DataFrame(data2)
        naive_df = pd.merge(df1, df2, on = 'id')
        naive_df = naive_df[naive_df['length'] >= plot_min_length]
        #print(naive_df)
        
        fig, ax = plt.subplots()
        ax.set_xlim([0, 100])
        cmap = plt.matplotlib.colormaps.get_cmap('plasma')
        
        sc = plt.scatter(naive_df['average_coverage'], naive_df['gc_content'],  c = naive_df['length'], cmap = cmap)
        cbar = plt.colorbar(sc)
        # Add labels and legend
        plt.xlabel('Average Coverage %')
        plt.ylabel('GC content %')
        plt.title('Blob plot with naive averages')
        plt.legend(loc="upper right")
        ax.legend([sc], ['Length'], loc="upper right")
        
        # save the plot
        save_path = os.path.join(out_dir, "Blob_naive.png")
        plt.savefig(save_path)
        plt.close()
        
    else :
        print("Visualize Naive Blob is turned off.")

    #---------------------------------------------------------------------------------------------------------------------------
    # Visualize Trimmed/chunked Blob
    #---------------------------------------------------------------------------------------------------------------------------

    if visualize_blob:
        
        data = [{'id': k, 'gc_content': v[0], 'length' : v[1]} for k, v in gc_trimmed_mean.items()]
        # Convert the list of dictionaries to dataframes
        df1 = pd.DataFrame(data)
        trimmed_df = pd.merge(df1, trimmed_mean, on = 'id')
        trimmed_df = trimmed_df[trimmed_df['length'] >= plot_min_length]
        #print(trimmed_df) 
        
        fig, ax = plt.subplots()
        ax.set_xlim([0, 100])
        cmap = plt.matplotlib.colormaps.get_cmap('plasma')
        
        #labels = list(trimmed_df['id'])
        sc1 = plt.scatter(trimmed_df['average_coverage'], trimmed_df['gc_content'], c = trimmed_df['length'], cmap = cmap)
        cbar = plt.colorbar(sc1)
        # Add labels and legend
        plt.xlabel('Trimmed Average Coverage %')
        plt.ylabel('GC content %')
        plt.title('Blob plot with trimmed averages with chunks of ' + str(chunk_size))
        ax.legend([sc1], ['Length'], loc="upper right")
        
        # save the plot
        save_path = os.path.join(out_dir, "Blob.png")
        plt.savefig(save_path)
        plt.close()

    else :
        print("Visualize Trimmed Blob is turned off.")    

    #---------------------------------------------------------------------------------------------------------------------------
    # Histogram of Contig Lengths
    #---------------------------------------------------------------------------------------------------------------------------

    if visualize_histogram:
        data = trimmed_df["length"]
        #print(data)
        
        fig, ax = plt.subplots()
        plt.hist(data, len(data))
        ax.ticklabel_format(style='plain', axis='x')
        plt.xlabel('Contig Lengths')
        plt.ylabel('Count')
        plt.title('Histogram of contig lengths')
        
        # save the plot
        save_path = os.path.join(out_dir, "Histogram_of_lengths.png")
        plt.savefig(save_path)
        plt.close()

    #---------------------------------------------------------------------------------------------------------------------------
    # Kmeans
    #---------------------------------------------------------------------------------------------------------------------------

    print('Using this plot you will need to select which cluster is the main cluster for the next step')
    #kmeans to get clusters in blob plots
    np.random.seed(42)
    
    coverage = trimmed_df['average_coverage']
    gc_content = trimmed_df['gc_content']
    ids = trimmed_df['id'].values
    
    #make a feature matrix
    X = np.column_stack((coverage, gc_content))
    
    #kmeans clusters = n_clusters
    kmeans = KMeans(n_clusters, random_state = 42, n_init='auto').fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    #create a dictionary for each cluster
    cluster_points = {i: [] for i in range(len(centroids))}
    
    #add the ids to each cluster in the dictionary
    for i, label in enumerate(labels):
        cluster_points[label].append(ids[i])
    
    #look up each point in dataframe to get length
    for cluster in cluster_points:
        total_cluster_length = 0
        
        #for each id for each cluster into a seperate file
        with open(out_dir + 'cluster_info/cluster_' + str(cluster) + '.txt', 'w') as out:
            for i in cluster_points[cluster]:
                #write ids into output
                out.write(i + '\n')
                
                #get length and total length
                length = trimmed_df.loc[trimmed_df['id'] == i, 'length'].iloc[0]
                total_cluster_length += length
        print('total length of cluster ' + str(cluster) + ' : ' + str(total_cluster_length))
        
    colors = ['aqua', 'lawngreen']
    #plotting
    plt.scatter(X[:,0], X[:,1], c = labels)
    plt.scatter(centroids[:,0], centroids[:,1], marker = '*', s = 100, c = colors)
    plt.xlabel('Coverage %')
    plt.ylabel('GC_content %')
    plt.title('K-means clustering | Coverage vs GC_content')  
        
    # save the plot
    save_path = os.path.join(out_dir, "Kmeans.png")
    plt.savefig(save_path) 
    plt.close()
        
    #---------------------------------------------------------------------------------------------------------------------------
    # Make a Fasta file with selected Headers
    #---------------------------------------------------------------------------------------------------------------------------

    i = 0 
    while i < n_clusters -1: 
        #for each id in cluster 0 get the sequence and write into a fasta file
        #we already have a fasta file with these ids, we just need to extract only the ones we are interested in
        with open(out_dir + 'cluster_info/cluster_' + str(i) + '.fasta', 'w') as fasta:
            with open(out_dir + 'cluster_info/cluster_' + str(i) + '.txt', 'r') as cluster:
                for line in cluster:
                    header = line.strip()
                    #write header
                    fasta.write('>' + header + '\n')
                    #before next header write line for sequence
                    #search for the header line in assembly which includes header
                    with open(assembly, 'r') as big:
                        found = False
                        for line in big:
                            if found:
                                sequence = line.strip()
                                found = False
                                break
                            if header in line:
                                found = True
                    big.close()
                    fasta.write(str(sequence) + '\n')
            
            cluster.close()
        fasta.close()
        i+=1

    #---------------------------------------------------------------------------------------------------------------------------
    # Split each entry into Fasta files
    #---------------------------------------------------------------------------------------------------------------------------
    
    #Split each entry in fasta file, i.e every 10k take 1k sample
    #for each entry in fasta file, if length > 10000 then take small chunk every 10000bp
    i = 0 
    while i < n_clusters - 1: 
        with open(out_dir + 'cluster_info/cluster_' + str(target_cluster) + '.fasta', 'r') as fasta:
            for line in fasta:
                if '>' in line:
                    #write in split_contigs
                    #print(line)
                    line = ''.join(line.split())
                    contig_name = str(line.replace('>', ''))
                else:
                    with open(out_dir + 'cluster_info/cluster_' + str(target_cluster) + '/split_contigs/' + contig_name + '.fasta', 'w') as contig_file:
                        #split into substrings of len_to_Blast every space_to_Blast
                        length = len(line)
                        #print(length)
                        if length > (space_to_Blast + len_to_Blast):
                            #loop
                            pos = 0
                            while pos < length:
                                start = pos
                                end = pos + len_to_Blast
                                #header = startpos endpos
                                #sequence = sequence
                                seq = (line[start:end] + '\n')
                                header = ('>' + str(start) + ' : ' + str(end) + '\n')
                                pos += space_to_Blast + len_to_Blast
                                contig_file.write(header)
                                contig_file.write(seq)
                        
                    contig_file.close()
                    
        fasta.close()
        i+=1

    #---------------------------------------------------------------------------------------------------------------------------
    #Blast with the generated Fasta files
    #---------------------------------------------------------------------------------------------------------------------------
    



    #---------------------------------------------------------------------------------------------------------------------------
    # Get concencus for each contig
    #---------------------------------------------------------------------------------------------------------------------------

    #TODO Tie output from BLAST section into thes var
    #TODO check if other dictionaries need to be added, if run on other groups

    cluster_0 = './100kb_subset_data/cluster_0/split_contigs/'
    cluster_1 = './100kb_subset_data/cluster_1/split_contigs/'
    clusters = [cluster_0, cluster_1]
    needed = []
    
    #dictionaries
    octocoral = ['Dendronephthya gigantea', 'Xenia sp. Carnegie-2017']
    dinoflagellates = ['Breviolum minutum Mf 1.05b.01','Symbiodinium microadriaticum','Symbiodinium sp. clade A Y106',
                       'Symbiodinium sp. clade C Y103','Symbiodinium kawagutii','Symbiodinium natans',
                       'Symbiodinium sp. CCMP2592','Symbiodinium sp. KB8','Symbiodinium sp. CCMP2456',
                       'Symbiodinium pilosum','Symbiodinium necroappetens','Cladocopium goreaui']
    
    #for each cluster
    for cluster in clusters:
        print(cluster)
        all_contigs = {}
        concencus_dict = {}
        #for file in each cluster
        for file in os.listdir(cluster):
            filename = file.split('.')[0]
            #count_dict has a count of number of each occurence of match
            count_dict = {'Breviolum minutum Mf 1.05b.01':0,'Symbiodinium microadriaticum':0,
                          'Symbiodinium sp. clade A Y106':0,'Symbiodinium sp. clade C Y103':0,'Symbiodinium kawagutii':0,
                          'Symbiodinium natans':0,'Symbiodinium sp. CCMP2592':0,'Symbiodinium sp. KB8':0,
                          'Symbiodinium sp. CCMP2456':0,'Symbiodinium pilosum':0,'Symbiodinium necroappetens':0,
                          'Cladocopium goreaui':0,'Dendronephthya gigantea':0, 'Xenia sp. Carnegie-2017': 0}
            if '.long.tab' in file:
                with open(cluster + file, 'r') as f:
                    #print(filename)
                    #get header number, get last column
                    for line in f:
                        line = line.split(sep = '\t')
                        #increase count for match
                        for entry in count_dict:
                            if entry in line[24].strip('\n'):
                                count_dict[entry]+=1
                all_contigs[filename] = count_dict
                
        #get count of octocorral and dinoflagellates
        #print(len(all_contigs))
        for p in all_contigs:
            o = 0
            d = 0
            for entry in all_contigs[p]:
                #print(entry)
                if entry in dinoflagellates:
                    d+=all_contigs[p][entry]
                if entry in octocoral:
                    o+=all_contigs[p][entry]
            
            #decide (octocoral/dinoflagellates/mixed/none)
            decision = 'none'
            if d == 0 and o >= 1:
                decision = 'octocoral'
            if d >= 1 and o ==0 :
                decision = 'dinoflagellates'
            if d >= 1 and o >= 1:
                if (d/o) < threshold:
                    decision = 'octo'
                    #print('octo '+str(d)+' '+str(o))
                else:
                    decision = 'mixed'
                    #print('mixed '+str(d)+' '+str(o))
            #add decision to concensus dict
            concencus_dict[p] = decision     
            
        #print(concencus_dict)
        
        #print totals of each
        octo=0
        dino=0
        mixed=0
        none=0
        
        for key in concencus_dict:
            if concencus_dict[key] in 'octocoral':
                octo +=1
                needed.append(key)
            if concencus_dict[key] in 'dinoflagellates':
                dino +=1
            if concencus_dict[key] in 'mixed':
                mixed +=1
            if concencus_dict[key] in 'none':
                none +=1
        print('octo: ' + str(octo))
        print('dino: ' + str(dino))
        print('mixed: ' + str(mixed))
        print('none: ' + str(none))

    #write to output file
    with open('./out_dir/cluster_info/needed.txt','w') as out:
        for i in needed:
            out.write(str(i) + '\n')
    out.close()
    #print(len(needed))

    #---------------------------------------------------------------------------------------------------------------------------
    #Blob plot with Fasta
    #---------------------------------------------------------------------------------------------------------------------------
    
    #trimmed_df['id'] = int(trimmed_df['id'])
    with open('out_dir/cluster_info/needed.txt','r') as keys:
        tot = []
        for line in keys:
            k = line.strip('\n')
            tot.append(k)
    print(len(tot))
        
    #plot blob again and color contigs with keys as labels
    data = [{'id': k, 'gc_content': v[0], 'length' : v[1]} for k, v in gc_trimmed_mean.items()]
    # Convert the list of dictionaries to dataframes
    df1 = pd.DataFrame(data)
    trimmed_df = pd.merge(df1, trimmed_mean, on = 'id')
    trimmed_df = trimmed_df[trimmed_df['length'] >= plot_min_length]
    #print(trimmed_df) 
    
    #add column if contained in keys
    trimmed_df['in_fasta'] = trimmed_df['id'].isin(tot)
    #print(trimmed_df)
    print(f"Fasta contigs found in this set{str(trimmed_df.in_fasta.sum())}")
    
    fig, ax = plt.subplots()
    ax.set_xlim([0, 100])
    sc1 = plt.scatter(trimmed_df['average_coverage'], trimmed_df['gc_content'], c = trimmed_df['in_fasta'])
    cbar = plt.colorbar(sc1)
    # Add labels and legend
    plt.xlabel('Trimmed Average Coverage %')
    plt.ylabel('GC content %')
    plt.title('Blob plot with trimmed averages with chunks of ' + str(chunk_size))
    ax.legend([sc1], ['in_fasta'], loc="upper right")
    # save the plot
    save_path = os.path.join(out_dir, "Fasta_Blob.png")
    plt.savefig(save_path)
    plt.close()

    #---------------------------------------------------------------------------------------------------------------------------
    # Histogram of the lengths of this subset
    #---------------------------------------------------------------------------------------------------------------------------
    
    #Bar plot of lengths of in fasta in trimmed_df
    data = trimmed_df[trimmed_df["in_fasta"] == True]
    data = data.sort_values(by=['length'], ascending=False)
    #print(data)
    #x is ids
    x = data['id']
    #y is length of ids
    y = data['length']
    
    #color contigs up to total length of genome size
    current = 0
    length = 430000000
    #add lengths together till greater than length
    for i, l in enumerate(y):
        current += l
        if current > length:
            cutoff_num = i
            cutoff = l
            break
            
    mask1 = y >= cutoff
    mask2 = y < cutoff
    
    plt.bar(x[mask1], y[mask1], color = 'red')
    plt.bar(x[mask2], y[mask2], color = 'blue')
    #plt.bar(x, y)
    
    plt.xlabel('Contig ids')
    plt.ylabel('Lengths')
    plt.title('Bar plot of contig lengths in Blast results')
    save_path = os.path.join(out_dir, "Histogram_of_subset.png")
    plt.savefig(save_path)
    plt.close()



    #TODO add step to count how many contigs are needed and replace the 190 here --done
    #TODO validate that cutoff_num works as intended
    #save selected contigs
    subset = data.head(cutoff_num)
    #print(subset)
    subset.to_csv('./out_dir/blast_subset.csv',index=False)

    #---------------------------------------------------------------------------------------------------------------------------
    # Add selected subset to fasta file
    #---------------------------------------------------------------------------------------------------------------------------

    #for each id in blast_subset add that contig to a fasta file
    ids = []
    with open('./out_dir/blast_subset.csv','r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            ids.append(row[0])
        
    output_file = open('ourGenome.fasta', 'a')
    with open(assembly, 'r') as fasta_file:
        header = ''
        sequence = ''
        for line in fasta_file:
            if line.startswith('>'):
                #check if id in list
                if any(id in line for id in ids):
                    header = line.strip()
            else:
                sequence = line.strip()
                if header and sequence:
                    output_file.write(header + '\n')
                    output_file.write(sequence + '\n')
                    header = ''
                    sequence = ''
                    
    output_file.close()

    #---------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------- 
if __name__ == '__main__':
    os.chdir("/netvolumes/srva229/molpal/hpc_exchange/Alex/cladorhizid_processing")
    app()
#-------------------------------------------------------------------------------------------------------------------------------
#TODO
#Need coverage to continue testing and finishing up the rest of this

#python filter_genome.py --assembly "./data/cladorhizid_v0.6_hapA/clado_v0.6_hapA.fasta" --coverage-per-contig "./data/heliopora_coerulea_hifi_2_p_assembly.coverage" --coverage-per-base "./data/heliopora_coerulea_hifi_2_p_assembly.depth" --n-cores 8



