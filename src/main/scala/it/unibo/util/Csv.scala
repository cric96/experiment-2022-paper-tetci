package it.unibo.util

import java.io.{File, PrintWriter}
import java.util.Locale
import scala.util.Using

/** Minimal CSV output for the custom exports read by the plotting scripts. */
object Csv {

  val Separator: String = ","

  /** `value` with `decimals` digits after the dot, independently of the default locale. */
  def number(value: Double, decimals: Int): String = s"%.${decimals}f".formatLocal(Locale.ROOT, value)

  /** Writes `header` and `rows` to `directory/name` (creating the directory if needed) and returns the file. */
  def write(directory: File, name: String, header: Seq[String], rows: Iterable[Seq[String]]): File = {
    directory.mkdirs()
    val file = new File(directory, name)
    Using.resource(new PrintWriter(file)) { writer =>
      writer.println(header.mkString(Separator))
      rows.foreach(row => writer.println(row.mkString(Separator)))
    }
    file
  }
}
