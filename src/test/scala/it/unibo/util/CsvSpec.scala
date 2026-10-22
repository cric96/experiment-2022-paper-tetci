package it.unibo.util

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.file.Files
import java.util.Locale
import scala.jdk.CollectionConverters._

class CsvSpec extends AnyFlatSpec with Matchers {

  private def withDefaultLocale[A](locale: Locale)(body: => A): A = {
    val original = Locale.getDefault
    Locale.setDefault(locale)
    try body
    finally Locale.setDefault(original)
  }

  "A number" should "be formatted with a dot as decimal separator, whatever the default locale" in {
    withDefaultLocale(Locale.ITALY) {
      Csv.number(3.14159, decimals = 2) shouldBe "3.14"
      Csv.number(-0.5f, decimals = 4) shouldBe "-0.5000"
    }
  }

  "A CSV file" should "contain the header followed by the rows, creating the missing directories" in {
    val directory = Files.createTempDirectory("csv-spec").resolve("nested")
    val file = Csv.write(directory.toFile, "data.csv", Seq("a", "b"), Seq(Seq("1", "2"), Seq("3", "4")))
    Files.readAllLines(file.toPath).asScala shouldBe Seq("a,b", "1,2", "3,4")
  }
}
