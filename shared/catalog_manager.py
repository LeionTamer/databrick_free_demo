from dataclasses import dataclass, field
from typing import Any, Dict, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

@dataclass
class CatalogManager:
    fqn: str
    schema: StructType
    spark: SparkSession = field(default_factory=lambda: SparkSession.builder.getOrCreate())

    _buffer: List[Dict[str, Any]] = field(default_factory=list, init=False)
    auto_commit_size: int = 25 # automatically commit when it reaches this len

    def __post_init__(self):
        if "id" not in self.schema.fieldNames():
            raise ValueError("Schema must contain an 'id' field.")

        if not self._table_exists():
            print(f"Table {self.fqn} does not exist. Creating...")
            empty_df = self.spark.createDataFrame([], self.schema) # noqa: SCPAP001
            empty_df.write.format("delta").saveAsTable(self.fqn)
            print(f"Table {self.fqn} created successfully.")

    def _table_exists(self) -> bool:
        try: 
            self.spark.catalog.tableExists(self.fqn)
            return True
        except Exception:
            return False
    
    def add_entry(self, entry: Dict[str, Any]):
        """
        Appends a single entry to the Delta table.
        Note: Frequent single-row inserts in Spark create small file issues.
        """
        self._buffer.append(entry)

        if len(self._buffer) >= self.auto_commit_size:
            self.commit()
    
    def commit(self):
        if not self._buffer:
            return
        
        df = self.spark.createDataFrame(self._buffer, self.schema) # noqa: SCPAP001
        df.write.format("delta").mode("append").saveAsTable(self.fqn)
        
        self._buffer.clear()

    def commit_dataframe(self, df: DataFrame):
        """
        Appends a PySpark DataFrame directly to the Delta table,
        validating the schema first.
        """
        if df.schema != self.schema:
            raise ValueError(
                f"Dataframe schema does not match the catalog schema.\n"
                f"Expected: {self.schema.simpleString()}\n"
                f"Got: {df.schema.simpleString()}"
            )

        if df.isEmpty():
            return
            
        df.write.format("delta").mode("append").saveAsTable(self.fqn)

    def commit_pandas_dataframe(self, pdf):
        """
        Converts a Pandas DataFrame to a Spark DataFrame and commits it.
        """
        # Skip empty pandas dataframes
        if pdf.empty:
            return
            
        # Convert pandas DataFrame to Spark DataFrame using the expected schema
        spark_df = self.spark.createDataFrame(pdf, schema=self.schema)
        
        # Reuse your existing Spark commit method
        self.commit_dataframe(spark_df)